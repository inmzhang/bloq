use std::ops::BitXor;

use bloq_utils::Pauli;
use glam::IVec2;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use crate::{CircuitError, CoordinateOverflowError, checked_translate_coordinate};

const PAULI_MAP_INLINE_CAP: usize = 4;
type PauliMapEntries = SmallVec<[(IVec2, Pauli); PAULI_MAP_INLINE_CAP]>;

/// A sparse, phaseless map from qubit coordinate to non-identity [`Pauli`].
/// Compose maps with `^`; exact product phases are discarded.
#[derive(Debug, PartialEq, Eq, Default, serde::Serialize)]
pub struct PauliMap {
    entries: PauliMapEntries,
}

// Manual impl: `find_index`'s binary search silently misbehaves when the
// entries are not strictly sorted by `(x, y)`, and every constructor upholds
// that plus the no-identity rule — so decoded (untrusted) entries must be
// checked here, not trusted from the wire. The wire struct mirrors the
// derived `Serialize` shape exactly.
impl<'de> serde::Deserialize<'de> for PauliMap {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(serde::Deserialize)]
        #[serde(rename = "PauliMap")]
        struct Wire {
            entries: PauliMapEntries,
        }

        let Wire { entries } = Wire::deserialize(deserializer)?;
        if !entries
            .windows(2)
            .all(|pair| (pair[0].0.x, pair[0].0.y) < (pair[1].0.x, pair[1].0.y))
        {
            return Err(serde::de::Error::custom(
                "PauliMap entries must be strictly sorted by (x, y)",
            ));
        }
        if entries.iter().any(|&(_, pauli)| pauli == Pauli::I) {
            return Err(serde::de::Error::custom(
                "PauliMap entries must not contain identity Paulis",
            ));
        }
        Ok(Self { entries })
    }
}

// Manual impl: `SmallVec`'s derived clone goes through a per-element iterator;
// `from_slice` copies the `Copy` entries in one shot. Cloning is hot when
// template chunks are translated to node coordinates.
impl Clone for PauliMap {
    fn clone(&self) -> Self {
        Self {
            entries: PauliMapEntries::from_slice(&self.entries),
        }
    }
}

// Manual impl: hashing is hot in flow matching, where maps key open-flow
// tables. Each entry is folded into a single `u64` write: x in the high half,
// y in the low half, and the 2-bit Pauli xor-mixed into y's top bits (real
// coordinates are far too small to reach bit 30, and even an aliased entry
// only weakens the hash, never its correctness).
impl std::hash::Hash for PauliMap {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.hash_translated(IVec2::ZERO, state);
    }
}

impl PauliMap {
    /// Hash the map as if translated by `offset`, without materializing the
    /// translated copy. Equal absolute maps hash equally regardless of how
    /// they are split into (entries, offset).
    pub(crate) fn hash_translated<H: std::hash::Hasher>(&self, offset: IVec2, state: &mut H) {
        state.write_usize(self.entries.len());
        for &(coord, pauli) in &self.entries {
            let coord = coord + offset;
            let packed = ((coord.x as u32 as u64) << 32) | (coord.y as u32 as u64);
            state.write_u64(packed ^ ((pauli as u64) << 30));
        }
    }
}

impl FromIterator<(IVec2, Pauli)> for PauliMap {
    fn from_iter<T: IntoIterator<Item = (IVec2, Pauli)>>(iter: T) -> Self {
        let mut map = Self::empty();
        for (coord, pauli) in iter {
            map.insert(coord, pauli);
        }
        map
    }
}

impl std::fmt::Display for PauliMap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_empty() {
            return write!(f, "I");
        }
        let mut entries = self.iter().collect::<Vec<_>>();
        entries.sort_unstable_by_key(|&(coord, _)| (coord.y, coord.x));
        for (index, (coord, pauli)) in entries.into_iter().enumerate() {
            if index > 0 {
                write!(f, "*")?;
            }
            write!(f, "{pauli}({},{})", coord.x, coord.y)?;
        }
        Ok(())
    }
}

impl PauliMap {
    /// Last entry strictly before `coord` in the map's (x, y) order.
    pub(crate) fn entry_before(&self, coord: IVec2) -> Option<(IVec2, Pauli)> {
        let end = self
            .entries
            .partition_point(|&(entry, _)| (entry.x, entry.y) < (coord.x, coord.y));
        end.checked_sub(1).map(|index| self.entries[index])
    }

    /// Exact power of `i` contributed by a product; the maps themselves remain phaseless.
    pub fn product_phase(&self, other: &Self) -> u8 {
        self.iter().fold(0, |phase, (coord, &left)| {
            (phase + left.product_phase(other.get(coord).copied().unwrap_or(Pauli::I))) % 4
        })
    }

    /// Returns a copy with every coordinate shifted by `offset`.
    ///
    /// # Panics
    ///
    /// Panics if any translated coordinate exceeds the signed 32-bit lattice.
    /// Use [`Self::try_translated`] for untrusted offsets.
    #[must_use]
    pub fn translated(&self, offset: IVec2) -> Self {
        self.try_translated(offset)
            .expect("translated PauliMap coordinates must fit in the i32 lattice")
    }

    /// Returns a copy with every coordinate shifted by `offset`.
    ///
    /// # Errors
    ///
    /// Returns [`CoordinateOverflowError`] if any translated coordinate exceeds
    /// the signed 32-bit lattice.
    pub fn try_translated(&self, offset: IVec2) -> Result<Self, CoordinateOverflowError> {
        // Translation by a constant offset preserves the coordinate sort
        // order, so the entries can be copied and shifted in place.
        let mut entries = PauliMapEntries::from_slice(&self.entries);
        for (coord, _) in &mut entries {
            *coord = checked_translate_coordinate(*coord, offset)?;
        }
        Ok(Self { entries })
    }

    /// Build a map from entries with pairwise-distinct coordinates and
    /// non-identity Paulis.
    ///
    /// Skips the per-entry binary search and shifting that `FromIterator`
    /// performs, sorting the collected entries once instead. The caller must
    /// guarantee the entries need no deduplication or identity filtering.
    ///
    /// # Panics
    ///
    /// Panics if coordinates repeat or an entry is identity.
    #[must_use]
    pub fn from_unique_entries(entries: impl IntoIterator<Item = (IVec2, Pauli)>) -> Self {
        let mut entries: PauliMapEntries = entries.into_iter().collect();
        entries.sort_unstable_by_key(|&(coord, _)| (coord.x, coord.y));
        assert!(
            entries
                .windows(2)
                .all(|pair| (pair[0].0.x, pair[0].0.y) < (pair[1].0.x, pair[1].0.y)),
            "entries must have pairwise-distinct coordinates"
        );
        assert!(
            entries.iter().all(|&(_, pauli)| pauli != Pauli::I),
            "entries must not contain identity Paulis"
        );
        Self { entries }
    }

    /// Creates an empty Pauli map.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            entries: SmallVec::new(),
        }
    }

    /// Returns the number of non-identity entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns whether the map contains no non-identity entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Iterates over the `(coordinate, pauli)` entries in sorted `(x, y)` order.
    pub fn iter(&self) -> PauliMapIter<'_> {
        PauliMapIter {
            inner: self.entries.iter(),
        }
    }

    /// The product's representative coordinate: its minimum data qubit by
    /// `(x, y)`, used as the [`crate::Op::MPP`] measurement-record key. Entries
    /// are kept sorted, so this is the first one. Recovering it from the product
    /// alone lets the construct/clone/translate/merge stages agree without a
    /// shared registry. `None` for an empty product.
    #[must_use]
    pub fn representative_coord(&self) -> Option<IVec2> {
        self.iter().next().map(|(coord, _)| *coord)
    }

    /// The Pauli at `coord`, or `None` if the coordinate carries identity.
    pub fn get(&self, coord: &IVec2) -> Option<&Pauli> {
        self.find_index(*coord)
            .ok()
            .map(|index| &self.entries[index].1)
    }

    /// Sets `coord` to `pauli`, returning the Pauli previously there.
    ///
    /// Inserting [`Pauli::I`] removes the entry, so the no-identity invariant
    /// holds.
    pub fn insert(&mut self, coord: IVec2, pauli: Pauli) -> Option<Pauli> {
        match self.find_index(coord) {
            Ok(index) => {
                if pauli == Pauli::I {
                    Some(self.entries.remove(index).1)
                } else {
                    Some(std::mem::replace(&mut self.entries[index].1, pauli))
                }
            }
            Err(index) => {
                if pauli != Pauli::I {
                    self.entries.insert(index, (coord, pauli));
                }
                None
            }
        }
    }

    /// Renders the map as a dense Pauli string of length `num_qubits`, placing
    /// each entry at its `layout` index and filling gaps with `_`.
    ///
    /// # Errors
    ///
    /// Returns [`CircuitError::QubitNotFoundInLayout`] if an entry's coordinate
    /// is absent from `layout`, or [`CircuitError::QubitIndexOutOfRange`] if its
    /// index is at least `num_qubits`.
    pub fn to_pauli_string(
        &self,
        layout: &FxHashMap<IVec2, u32>,
        num_qubits: usize,
    ) -> Result<String, CircuitError> {
        let mut pauli_string = "_".repeat(num_qubits);
        for (coord, pauli) in self {
            let index = *layout
                .get(coord)
                .ok_or(CircuitError::QubitNotFoundInLayout(*coord))?
                as usize;
            if index >= num_qubits {
                return Err(CircuitError::QubitIndexOutOfRange { index, num_qubits });
            }
            pauli_string.replace_range(
                index..index + 1,
                match pauli {
                    Pauli::I => "I",
                    Pauli::X => "X",
                    Pauli::Y => "Y",
                    Pauli::Z => "Z",
                },
            );
        }
        Ok(pauli_string)
    }

    fn find_index(&self, coord: IVec2) -> Result<usize, usize> {
        self.entries
            .binary_search_by_key(&(coord.x, coord.y), |(entry, _)| (entry.x, entry.y))
    }
}

impl BitXor for &PauliMap {
    type Output = PauliMap;

    // Both maps are sorted, so composition is one linear merge.
    fn bitxor(self, rhs: Self) -> Self::Output {
        let (a, b) = (&self.entries, &rhs.entries);
        let mut entries = PauliMapEntries::with_capacity(a.len() + b.len());
        let (mut i, mut j) = (0, 0);
        while i < a.len() && j < b.len() {
            let ((a_coord, a_pauli), (b_coord, b_pauli)) = (a[i], b[j]);
            match (a_coord.x, a_coord.y).cmp(&(b_coord.x, b_coord.y)) {
                std::cmp::Ordering::Less => {
                    entries.push((a_coord, a_pauli));
                    i += 1;
                }
                std::cmp::Ordering::Greater => {
                    entries.push((b_coord, b_pauli));
                    j += 1;
                }
                std::cmp::Ordering::Equal => {
                    let pauli = a_pauli ^ b_pauli;
                    if pauli != Pauli::I {
                        entries.push((a_coord, pauli));
                    }
                    i += 1;
                    j += 1;
                }
            }
        }
        entries.extend_from_slice(&a[i..]);
        entries.extend_from_slice(&b[j..]);
        PauliMap { entries }
    }
}

/// Borrowing iterator over a [`PauliMap`]'s sorted `(coordinate, pauli)`
/// entries.
#[derive(Debug)]
pub struct PauliMapIter<'a> {
    inner: std::slice::Iter<'a, (IVec2, Pauli)>,
}

impl<'a> Iterator for PauliMapIter<'a> {
    type Item = (&'a IVec2, &'a Pauli);

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|(coord, pauli)| (coord, pauli))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl ExactSizeIterator for PauliMapIter<'_> {}

impl std::iter::FusedIterator for PauliMapIter<'_> {}

impl DoubleEndedIterator for PauliMapIter<'_> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.inner.next_back().map(|(coord, pauli)| (coord, pauli))
    }
}

impl IntoIterator for PauliMap {
    type Item = (IVec2, Pauli);
    type IntoIter = smallvec::IntoIter<[(IVec2, Pauli); PAULI_MAP_INLINE_CAP]>;

    fn into_iter(self) -> Self::IntoIter {
        self.entries.into_iter()
    }
}

impl<'a> IntoIterator for &'a PauliMap {
    type Item = (&'a IVec2, &'a Pauli);
    type IntoIter = PauliMapIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map(entries: impl IntoIterator<Item = (IVec2, Pauli)>) -> PauliMap {
        entries.into_iter().collect()
    }

    /// Decoded entries are untrusted: unsorted or identity entries would
    /// silently break `find_index`'s binary search, so they must be rejected.
    #[test]
    fn deserialize_rejects_unsorted_and_identity_entries() {
        #[derive(serde::Serialize)]
        #[serde(rename = "PauliMap")]
        struct RawMap {
            entries: Vec<(IVec2, Pauli)>,
        }
        let encode = |entries: Vec<(IVec2, Pauli)>| {
            postcard::to_allocvec(&RawMap { entries }).expect("test wire encodes into a Vec")
        };

        let unsorted = encode(vec![
            (IVec2::new(1, 0), Pauli::X),
            (IVec2::new(0, 0), Pauli::Z),
        ]);
        postcard::from_bytes::<PauliMap>(&unsorted).unwrap_err();

        let identity = encode(vec![(IVec2::new(0, 0), Pauli::I)]);
        postcard::from_bytes::<PauliMap>(&identity).unwrap_err();

        let valid = map([(IVec2::new(0, 0), Pauli::X), (IVec2::new(1, 0), Pauli::Z)]);
        let bytes = postcard::to_allocvec(&valid).expect("map encodes into a Vec");
        let restored: PauliMap = postcard::from_bytes(&bytes).expect("sorted entries decode");
        assert_eq!(restored, valid);
    }

    #[test]
    fn unique_entries_preserve_map_invariants() {
        for entries in [
            [(IVec2::ZERO, Pauli::X), (IVec2::ZERO, Pauli::Z)],
            [(IVec2::ZERO, Pauli::X), (IVec2::X, Pauli::I)],
        ] {
            std::panic::catch_unwind(|| PauliMap::from_unique_entries(entries))
                .expect_err("duplicate coordinates and identity entries must panic");
        }
        let map = PauliMap::from_unique_entries([(IVec2::X, Pauli::Z), (IVec2::ZERO, Pauli::X)]);
        let mut entries: crate::PauliMapIter<'_> = map.iter();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries.next(), Some((&IVec2::ZERO, &Pauli::X)));
        assert_eq!(entries.len(), 1);
        assert_eq!(entries.next_back(), Some((&IVec2::X, &Pauli::Z)));
        assert_eq!(entries.len(), 0);
        assert_eq!(entries.next(), None);
    }

    #[test]
    fn xor_composes_paulis() {
        let x = map([(IVec2::ZERO, Pauli::X)]);
        let z = map([(IVec2::ZERO, Pauli::Z)]);

        assert_eq!(&x ^ &z, map([(IVec2::ZERO, Pauli::Y)]));
        assert!((&x ^ &x).is_empty());
    }

    #[test]
    fn pauli_string_error_reports_empty_layout_size() {
        let coord = IVec2::ZERO;
        let map = map([(coord, Pauli::X)]);
        let layout: FxHashMap<_, _> = [(coord, 0)].into_iter().collect();

        assert_eq!(
            map.to_pauli_string(&layout, 0),
            Err(CircuitError::QubitIndexOutOfRange {
                index: 0,
                num_qubits: 0,
            })
        );
    }

    #[test]
    fn pauli_string_uses_layout_indices_and_identity_gaps() {
        let map = map([
            (IVec2::ZERO, Pauli::Z),
            (IVec2::X, Pauli::X),
            (IVec2::Y, Pauli::Y),
        ]);
        let layout = FxHashMap::from_iter([(IVec2::ZERO, 3), (IVec2::X, 0), (IVec2::Y, 2)]);
        assert_eq!(map.to_pauli_string(&layout, 5), Ok("X_YZ_".into()));
        assert_eq!(
            map.to_pauli_string(&FxHashMap::default(), 5),
            Err(CircuitError::QubitNotFoundInLayout(IVec2::ZERO))
        );
        assert_eq!(
            PauliMap::empty().to_pauli_string(&layout, 0),
            Ok(String::new())
        );
    }

    #[test]
    fn checked_translation_rejects_coordinate_overflow() {
        let map = map([(IVec2::new(i32::MAX, 0), Pauli::X)]);

        assert_eq!(
            map.try_translated(IVec2::X),
            Err(CoordinateOverflowError {
                coordinate: IVec2::new(i32::MAX, 0),
                offset: IVec2::X,
            })
        );
    }

    #[test]
    fn pauli_map_deduplicates_and_sorts_entries() {
        let map = vec![
            (IVec2::new(2, 0), Pauli::Z),
            (IVec2::new(0, 1), Pauli::X),
            (IVec2::new(2, 0), Pauli::I),
            (IVec2::new(1, 0), Pauli::Y),
        ]
        .into_iter()
        .collect::<PauliMap>();

        assert_eq!(
            map.into_iter().collect::<Vec<_>>(),
            vec![(IVec2::new(0, 1), Pauli::X), (IVec2::new(1, 0), Pauli::Y)]
        );
    }
}
