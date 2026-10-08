use std::ops::{BitAnd, BitOr, BitXor, BitXorAssign};

use binar::{BitVec, Bitwise, BitwiseMut, BitwisePair, BitwisePairMut};
use strum::{Display, EnumString};
use thiserror::Error;

/// Error returned by the fallible conversions into [`Pauli`] and [`Basis`].
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum PauliError {
    /// The [`Pauli`] is `I` or `Y`, which have no [`Basis`] counterpart.
    #[error("cannot convert Pauli {0} to Basis")]
    NotBasis(Pauli),
    /// The byte is outside the `0..=3` range that encodes a [`Pauli`].
    #[error("invalid u8 value for Pauli: {0}")]
    InvalidByte(u8),
    /// The character is not one of `I`, `_`, `X`, `Y`, `Z`.
    #[error("invalid char for Pauli: {0}")]
    InvalidChar(char),
}

/// Pauli `X` or `Z` basis (excludes `Y` and identity).
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    PartialOrd,
    Ord,
    Display,
    EnumString,
    serde::Serialize,
    serde::Deserialize,
)]
pub enum Basis {
    /// X basis.
    X,
    /// Z basis.
    Z,
}

impl Basis {
    /// Returns the complementary basis (`X` <-> `Z`).
    pub const fn flip(self) -> Self {
        match self {
            Self::X => Self::Z,
            Self::Z => Self::X,
        }
    }
}

impl TryFrom<Pauli> for Basis {
    type Error = PauliError;

    fn try_from(value: Pauli) -> Result<Self, Self::Error> {
        match value {
            Pauli::X => Ok(Basis::X),
            Pauli::Z => Ok(Basis::Z),
            _ => Err(PauliError::NotBasis(value)),
        }
    }
}

/// Pauli basis: `X`, `Y`, or `Z` (excludes identity).
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    EnumString,
    Display,
    serde::Serialize,
    serde::Deserialize,
)]
pub enum PauliBasis {
    /// X basis.
    X,
    /// Y basis.
    Y,
    /// Z basis.
    Z,
}

impl From<Basis> for PauliBasis {
    fn from(basis: Basis) -> Self {
        match basis {
            Basis::X => PauliBasis::X,
            Basis::Z => PauliBasis::Z,
        }
    }
}

/// A single-qubit Pauli operator.
///
/// The discriminants are the XZ symplectic encoding (`X` sets bit 0, `Z` sets
/// bit 1), so the bit algebra in [`BitOr`]/[`BitXor`]/[`BitAnd`] and
/// [`PauliString`] depends on these exact values. Keep them fixed.
#[repr(u8)]
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    EnumString,
    Display,
    serde::Serialize,
    serde::Deserialize,
)]
pub enum Pauli {
    /// Identity.
    I = 0,
    /// Pauli X.
    X = 1,
    /// Pauli Z.
    Z = 2,
    /// Pauli Y.
    Y = 3,
}

impl From<Basis> for Pauli {
    fn from(basis: Basis) -> Self {
        match basis {
            Basis::X => Pauli::X,
            Basis::Z => Pauli::Z,
        }
    }
}

impl From<PauliBasis> for Pauli {
    fn from(pb: PauliBasis) -> Self {
        match pb {
            PauliBasis::X => Pauli::X,
            PauliBasis::Y => Pauli::Y,
            PauliBasis::Z => Pauli::Z,
        }
    }
}

impl Pauli {
    const fn from_xz(x: bool, z: bool) -> Self {
        match (x, z) {
            (false, false) => Self::I,
            (true, false) => Self::X,
            (false, true) => Self::Z,
            (true, true) => Self::Y,
        }
    }

    /// Swaps `X` and `Z`; leaves `I` and `Y` unchanged.
    pub const fn flip(self) -> Self {
        match self {
            Self::X => Self::Z,
            Self::Z => Self::X,
            _ => self,
        }
    }

    /// Iterates the `X` and `Z` components present in this Pauli.
    pub fn iter_xz(self) -> impl Iterator<Item = Self> {
        [Self::X, Self::Z].into_iter().filter(move |&p| p & self)
    }

    /// Two Paulis anticommute iff both are non-identity and differ.
    pub const fn anticommutes(self, other: Self) -> bool {
        !matches!(self, Self::I) && !matches!(other, Self::I) && self as u8 != other as u8
    }
}

impl BitOr for Pauli {
    type Output = Pauli;

    fn bitor(self, rhs: Self) -> Self::Output {
        let int_or = self as u8 | rhs as u8;
        Pauli::try_from(int_or).expect("OR of two Paulis is always a valid Pauli")
    }
}

impl BitXor for Pauli {
    type Output = Pauli;

    fn bitxor(self, rhs: Self) -> Self::Output {
        let int_xor = self as u8 ^ rhs as u8;
        Pauli::try_from(int_xor).expect("XOR of two Paulis is always a valid Pauli")
    }
}

impl BitAnd for Pauli {
    type Output = bool;

    fn bitand(self, rhs: Self) -> Self::Output {
        let int_and = self as u8 & rhs as u8;
        int_and != 0
    }
}

impl TryFrom<u8> for Pauli {
    type Error = PauliError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Pauli::I),
            1 => Ok(Pauli::X),
            2 => Ok(Pauli::Z),
            3 => Ok(Pauli::Y),
            _ => Err(PauliError::InvalidByte(value)),
        }
    }
}

impl From<Pauli> for u8 {
    fn from(value: Pauli) -> Self {
        value as u8
    }
}

impl TryFrom<char> for Pauli {
    type Error = PauliError;

    fn try_from(value: char) -> Result<Self, Self::Error> {
        match value {
            'I' | '_' => Ok(Pauli::I),
            'X' => Ok(Pauli::X),
            'Y' => Ok(Pauli::Y),
            'Z' => Ok(Pauli::Z),
            _ => Err(PauliError::InvalidChar(value)),
        }
    }
}

/// A dense unsigned Pauli operator over `len` qubits, stored as XZ bit vectors.
///
/// Phaseless, with each site read as the *Hermitian* single-qubit Pauli it
/// names — a site carrying both bits is `Y = iXZ`, not the bare product `XZ`.
/// A product of Hermitian factors on distinct sites is Hermitian, so every
/// `PauliString` is a valid measurement observable or rotation axis by
/// construction; there is no phase to make it otherwise.
/// Operations between different widths treat missing trailing sites as
/// identities; XOR keeps the wider width.
/// Parse strings of `I`, `_`, `X`, `Y`, and `Z` with [`str::parse`];
/// [`Display`](std::fmt::Display) renders identity as `_`.
#[derive(Clone, PartialEq, Eq)]
pub struct PauliString {
    xs: BitVec,
    zs: BitVec,
}

/// A packed Hermitian [`PauliString`] with an exact phase `i^k`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhasedPauliString {
    /// Hermitian body. Mutating it does not change the phase.
    pub paulis: PauliString,
    phase: u8,
}

impl PhasedPauliString {
    /// Attaches `i^phase` to `paulis`, reducing the exponent modulo four.
    pub fn new(paulis: PauliString, phase: u8) -> Self {
        Self {
            paulis,
            phase: phase & 3,
        }
    }

    /// Creates `+paulis`.
    pub fn positive(paulis: PauliString) -> Self {
        Self::new(paulis, 0)
    }

    /// Returns the phase exponent in `0..=3`.
    pub const fn phase(&self) -> u8 {
        self.phase
    }

    /// Adds a phase exponent modulo four.
    pub fn shift_phase(&mut self, exponent: u8) {
        self.phase = self.phase.wrapping_add(exponent) & 3;
    }

    /// Multiplies by `other`, retaining the exact phase.
    pub fn multiply_assign(&mut self, other: &Self) {
        self.shift_phase(other.phase + self.paulis.product_phase(&other.paulis));
        self.paulis ^= &other.paulis;
    }

    /// Conjugates one column by Hadamard, including `HYH = -Y`.
    pub fn conjugate_h(&mut self, column: usize) {
        let pauli = self.paulis.get(column);
        self.paulis.set(column, pauli.flip());
        if pauli == Pauli::Y {
            self.shift_phase(2);
        }
    }

    /// Returns the sign of a Hermitian operator (`true` means negative).
    pub const fn hermitian_sign(&self) -> Option<bool> {
        match self.phase {
            0 => Some(false),
            2 => Some(true),
            _ => None,
        }
    }
}

impl PauliString {
    /// Creates the identity Pauli string over `num_qubits` qubits.
    pub fn new(num_qubits: usize) -> Self {
        Self {
            xs: BitVec::zeros(num_qubits),
            zs: BitVec::zeros(num_qubits),
        }
    }

    /// Creates the string with `pauli` at `index` and identity everywhere else.
    ///
    /// # Panics
    ///
    /// Panics if `index >= num_qubits`.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloq_utils::{Pauli, PauliString};
    ///
    /// let observable = PauliString::single(4, 2, Pauli::Y);
    /// assert_eq!(observable.to_string(), "__Y_");
    /// ```
    pub fn single(num_qubits: usize, index: usize, pauli: Pauli) -> Self {
        let mut paulis = PauliString::new(num_qubits);
        paulis.set(index, pauli);
        paulis
    }

    /// Collects `(index, pauli)` terms into a string over `num_qubits` qubits.
    ///
    /// Terms are *assignments*, not factors: a repeated index keeps the last
    /// Pauli given for it rather than multiplying the two.
    ///
    /// # Panics
    ///
    /// Panics if any index is `>= num_qubits`.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloq_utils::{Pauli, PauliString};
    ///
    /// let stabilizer = PauliString::from_terms(3, [(0, Pauli::Z), (2, Pauli::Z)]);
    /// assert_eq!(stabilizer.to_string(), "Z_Z");
    /// ```
    pub fn from_terms(num_qubits: usize, terms: impl IntoIterator<Item = (usize, Pauli)>) -> Self {
        let mut paulis = PauliString::new(num_qubits);
        for (index, pauli) in terms {
            paulis.set(index, pauli);
        }
        paulis
    }

    /// Returns `true` if `self` and `other` commute as operators.
    ///
    /// Two Pauli strings anticommute exactly when they anticommute on an odd
    /// number of sites, which the symplectic form `⟨x, other.z⟩ + ⟨z, other.x⟩`
    /// counts mod 2.
    ///
    /// The two strings need not name the same number of qubits. Where one runs
    /// out the other is being compared against implicit identities, which
    /// commute with everything — so the fold below stops at the shorter operand
    /// rather than requiring equal widths. Going through the words directly is
    /// also what avoids `BitVec::dot`, whose length assertion holds in release
    /// builds and would turn a width mismatch into a panic.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloq_utils::PauliString;
    ///
    /// let xx = PauliString::try_from("XX")?;
    /// let zz = PauliString::try_from("ZZ")?;
    /// let zi = PauliString::try_from("Z_")?;
    /// assert!(xx.commutes_with(&zz));   // two anticommuting sites
    /// assert!(!xx.commutes_with(&zi));  // one
    ///
    /// // Widths may differ; the shorter string ends in implicit identities.
    /// let x_wide = PauliString::try_from("X___")?;
    /// assert!(!x_wide.commutes_with(&PauliString::try_from("Z")?));
    /// assert!(x_wide.commutes_with(&PauliString::try_from("_Z")?));
    /// # Ok::<(), bloq_utils::PauliError>(())
    /// ```
    pub fn commutes_with(&self, other: &Self) -> bool {
        /// Parity of `⟨a, b⟩` over the words the two operands share.
        fn dot_parity(a: &[u64], b: &[u64]) -> bool {
            a.iter().zip(b).fold(false, |parity, (&left, &right)| {
                parity ^ ((left & right).count_ones() % 2 != 0)
            })
        }
        dot_parity(self.x_words(), other.z_words()) == dot_parity(self.z_words(), other.x_words())
    }

    /// Returns the power of `i` contributed by multiplying these strings.
    /// Missing sites are identities.
    pub fn product_phase(&self, other: &Self) -> u8 {
        self.x_words()
            .iter()
            .zip(self.z_words())
            .zip(other.x_words().iter().zip(other.z_words()))
            .fold(0, |phase, ((&ax, &az), (&bx, &bz))| {
                let (ax_only, ay, az_only) = (ax & !az, ax & az, az & !ax);
                let (bx_only, by, bz_only) = (bx & !bz, bx & bz, bz & !bx);
                let positive =
                    ((ax_only & by) | (ay & bz_only) | (az_only & bx_only)).count_ones() as u8;
                let negative =
                    ((ay & bx_only) | (az_only & by) | (ax_only & bz_only)).count_ones() as u8;
                phase.wrapping_add(positive).wrapping_sub(negative) & 3
            })
    }

    /// The `X` components as `u64` words, least-significant qubit first.
    ///
    /// Word-level access for bit algebra over whole strings — the simulator's
    /// frame decomposition walks these directly rather than one site at a time.
    /// Bits at or beyond [`len`](Self::len) are always clear.
    pub fn x_words(&self) -> &[u64] {
        self.xs.as_words()
    }

    /// The `Z` components as `u64` words. See [`x_words`](Self::x_words).
    pub fn z_words(&self) -> &[u64] {
        self.zs.as_words()
    }

    /// Returns the number of qubits.
    pub fn len(&self) -> usize {
        self.xs.len()
    }

    /// Returns `true` if there are no qubits.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the number of non-identity Paulis.
    pub fn weight(&self) -> usize {
        // A column is non-identity iff either axis bit is set, so the weight is
        // the popcount of `xs | zs`. Bits past `len` are never set.
        self.xs.or_weight(&self.zs)
    }

    /// Returns whether every site is identity, stopping at the first nonzero word.
    pub fn is_identity(&self) -> bool {
        self.xs.is_zero() && self.zs.is_zero()
    }

    /// Iterates the non-identity sites in ascending index order.
    ///
    /// # Panics
    ///
    /// Panics if the packed X/Z representation contains an impossible support bit.
    pub fn iter_support(&self) -> impl Iterator<Item = (usize, Pauli)> + '_ {
        self.x_words()
            .iter()
            .copied()
            .zip(self.z_words().iter().copied())
            .enumerate()
            .flat_map(|(word_index, (xs, zs))| {
                let mut support = xs | zs;
                std::iter::from_fn(move || {
                    if support == 0 {
                        return None;
                    }
                    let bit_index = support.trailing_zeros() as usize;
                    support &= support - 1;
                    let bit = 1 << bit_index;
                    let pauli = match (xs & bit != 0, zs & bit != 0) {
                        (true, false) => Pauli::X,
                        (false, true) => Pauli::Z,
                        (true, true) => Pauli::Y,
                        (false, false) => unreachable!("support bit has an X or Z component"),
                    };
                    Some((word_index * u64::BITS as usize + bit_index, pauli))
                })
            })
    }

    /// Returns the Pauli at `index`.
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of bounds.
    pub fn get(&self, index: usize) -> Pauli {
        assert!(
            index < self.len(),
            "index {index} out of bounds for PauliString of length {}",
            self.len()
        );
        Pauli::from_xz(self.xs.index(index), self.zs.index(index))
    }

    /// Sets the Pauli at `index`.
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of bounds.
    pub fn set(&mut self, index: usize, pauli: Pauli) {
        assert!(
            index < self.len(),
            "index {index} out of bounds for PauliString of length {}",
            self.len()
        );
        self.xs.assign_index(index, pauli as u8 & 1 != 0);
        self.zs.assign_index(index, pauli as u8 & 2 != 0);
    }
}

impl std::fmt::Debug for PauliString {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PauliString({self})")
    }
}

impl std::fmt::Display for PauliString {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for index in 0..self.len() {
            f.write_str(match self.get(index) {
                Pauli::I => "_",
                Pauli::X => "X",
                Pauli::Y => "Y",
                Pauli::Z => "Z",
            })?;
        }
        Ok(())
    }
}

impl std::str::FromStr for PauliString {
    type Err = PauliError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::try_from(value)
    }
}

impl TryFrom<&str> for PauliString {
    type Error = PauliError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        let num_qubits = value.len();
        let mut ps = PauliString::new(num_qubits);
        for (i, c) in value.chars().enumerate() {
            let p = Pauli::try_from(c)?;
            ps.set(i, p);
        }
        Ok(ps)
    }
}

impl BitXor for &PauliString {
    type Output = PauliString;

    fn bitxor(self, rhs: Self) -> Self::Output {
        let mut result = self.clone();
        result ^= rhs;
        result
    }
}

impl BitXorAssign<&PauliString> for PauliString {
    fn bitxor_assign(&mut self, other: &Self) {
        // Missing trailing sites are identities, so pad whichever operand is
        // shorter before using BitVec's equal-width operation.
        if self.len() > other.len() {
            let (mut xs, mut zs) = (other.xs.clone(), other.zs.clone());
            xs.resize(self.len());
            zs.resize(self.len());
            self.xs.bitxor_assign(&xs);
            self.zs.bitxor_assign(&zs);
            return;
        }
        if self.len() < other.len() {
            self.xs.resize(other.len());
            self.zs.resize(other.len());
        }
        self.xs.bitxor_assign(&other.xs);
        self.zs.bitxor_assign(&other.zs);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parsing_and_indexing_preserve_the_declared_width() {
        let paulis: PauliString = "IXYZ_".parse().unwrap();
        assert_eq!(paulis.to_string(), "_XYZ_");
        assert_eq!(paulis.to_string().parse::<PauliString>().unwrap(), paulis);
        assert_eq!(
            "Xλ".parse::<PauliString>(),
            Err(PauliError::InvalidChar('λ'))
        );

        // Index 1 is inside the backing word, but outside this string.
        let mut single = PauliString::new(1);
        std::panic::catch_unwind(|| single.get(1)).expect_err("out-of-width reads must panic");
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            single.set(1, Pauli::X);
        }))
        .expect_err("out-of-width writes must panic");
        assert_eq!(single.iter_support().count(), 0);
        assert_eq!(single.weight(), 0);
    }

    #[test]
    fn support_statistics_cross_word_boundaries() {
        // 70 columns spans two u64 blocks; X, Y, and Z must each count once.
        let mut ps = PauliString::new(70);
        for (index, pauli) in [
            (0, Pauli::X),
            (63, Pauli::Y),
            (64, Pauli::Z),
            (69, Pauli::X),
        ] {
            ps.set(index, pauli);
        }
        assert_eq!(ps.weight(), 4);
        ps.set(63, Pauli::I);
        assert_eq!(ps.weight(), 3);
        assert_eq!(PauliString::new(70).weight(), 0);
        for len in [0, 70] {
            assert!(PauliString::new(len).is_identity());
        }
        for pauli in [Pauli::X, Pauli::Y, Pauli::Z] {
            assert!(!PauliString::single(70, 64, pauli).is_identity());
        }
    }

    #[test]
    fn iter_support_skips_identities_across_word_boundaries() {
        let ps = PauliString::from_terms(
            130,
            [
                (0, Pauli::X),
                (63, Pauli::Y),
                (64, Pauli::Z),
                (129, Pauli::X),
            ],
        );
        assert_eq!(
            ps.iter_support().collect::<Vec<_>>(),
            [
                (0, Pauli::X),
                (63, Pauli::Y),
                (64, Pauli::Z),
                (129, Pauli::X)
            ]
        );
    }

    /// Operands of different widths used to reach `BitVec::dot`, whose length
    /// assertion fires in release builds — so a commutation test between a
    /// 4-qubit and a 6-qubit string aborted instead of answering. The shorter
    /// string's missing sites are identities and commute with anything.
    #[test]
    fn commutation_spans_operands_of_different_widths() {
        let narrow = PauliString::single(4, 0, Pauli::X);
        let wide = PauliString::single(6, 0, Pauli::Z);
        assert!(!narrow.commutes_with(&wide), "one anticommuting site");
        assert!(!wide.commutes_with(&narrow), "and symmetrically");

        // Only the sites the two share can anticommute.
        let disjoint = PauliString::single(6, 5, Pauli::Z);
        assert!(narrow.commutes_with(&disjoint));
        assert!(disjoint.commutes_with(&narrow));

        // Across a word boundary, where the wider operand has words the
        // narrower one does not.
        let short = PauliString::single(64, 63, Pauli::X);
        let long = PauliString::from_terms(200, [(63, Pauli::Z), (150, Pauli::X)]);
        assert!(!short.commutes_with(&long));
        assert!(!long.commutes_with(&short));

        // Two anticommuting sites commute overall, whatever the widths.
        let pair = PauliString::from_terms(4, [(0, Pauli::X), (1, Pauli::X)]);
        let pair_wide = PauliString::from_terms(9, [(0, Pauli::Z), (1, Pauli::Z)]);
        assert!(pair.commutes_with(&pair_wide));
    }

    #[test]
    fn xor_extends_the_shorter_operand_with_identities() {
        let short = PauliString::try_from("X").unwrap();
        let wide = PauliString::try_from("_Z").unwrap();

        assert_eq!((&short ^ &wide).to_string(), "XZ");
        assert_eq!(&short ^ &wide, &wide ^ &short);
    }

    #[test]
    fn packed_product_phase_matches_sitewise_pauli_products() {
        fn sitewise(left: &PauliString, right: &PauliString) -> u8 {
            (0..left.len().min(right.len())).fold(0, |phase, index| {
                (phase
                    + match (left.get(index), right.get(index)) {
                        (Pauli::X, Pauli::Y) | (Pauli::Y, Pauli::Z) | (Pauli::Z, Pauli::X) => 1,
                        (Pauli::Y, Pauli::X) | (Pauli::Z, Pauli::Y) | (Pauli::X, Pauli::Z) => 3,
                        _ => 0,
                    })
                    % 4
            })
        }

        let left = PauliString::from_terms(
            130,
            (0..130).map(|index| (index, [Pauli::I, Pauli::X, Pauli::Y, Pauli::Z][index % 4])),
        );
        let right = PauliString::from_terms(
            70,
            (0..70).map(|index| (index, [Pauli::Z, Pauli::Y, Pauli::X, Pauli::I][index % 4])),
        );
        assert_eq!(left.product_phase(&right), sitewise(&left, &right));
        assert_eq!(right.product_phase(&left), sitewise(&right, &left));
    }

    #[test]
    fn phased_pauli_retains_exact_products_and_clifford_signs() {
        let positive = |value| PhasedPauliString::positive(PauliString::try_from(value).unwrap());

        let mut product = positive("X");
        product.multiply_assign(&positive("Z"));
        assert_eq!(product.paulis.to_string(), "Y");
        assert_eq!(product.phase(), 3);

        let mut y = positive("Y");
        y.conjugate_h(0);
        assert_eq!(y.paulis.to_string(), "Y");
        assert_eq!(y.hermitian_sign(), Some(true));
    }
}
