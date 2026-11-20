use bloq_utils::PauliBasis;
use strum::{Display, EnumIter, EnumString, IntoEnumIterator, IntoStaticStr};

/// A quantum gate, named after its Stim mnemonic.
///
/// Variant names follow Stim's gate set. `strum` aliases map the common
/// synonyms (e.g. `CNOT` and `ZCX` both parse to [`GateType::CX`], and `R`
/// renders/parses [`GateType::RZ`]).
#[expect(
    non_camel_case_types,
    reason = "variants preserve canonical Stim gate mnemonics"
)]
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    EnumIter,
    EnumString,
    Display,
    IntoStaticStr,
    PartialOrd,
    Ord,
    serde::Serialize,
    serde::Deserialize,
)]
pub enum GateType {
    // Reset gates
    /// X-basis reset.
    RX,
    /// Y-basis reset.
    RY,
    /// Z-basis reset (`R` in Stim text).
    #[strum(serialize = "RZ", to_string = "R")]
    RZ,
    // Controlled gates
    /// X-controlled X.
    XCX,
    /// X-controlled Y.
    XCY,
    /// X-controlled Z.
    XCZ,
    /// Y-controlled X.
    YCX,
    /// Y-controlled Y.
    YCY,
    /// Y-controlled Z.
    YCZ,
    /// Z-controlled X.
    #[strum(to_string = "CX", serialize = "ZCX", serialize = "CNOT")]
    CX,
    /// Z-controlled Y.
    #[strum(to_string = "CY", serialize = "ZCY")]
    CY,
    /// Z-controlled Z.
    #[strum(to_string = "CZ", serialize = "ZCZ")]
    CZ,
    // Hadamard-like gates
    /// Exchanges X and Z axes.
    #[strum(to_string = "H", serialize = "H_XZ")]
    H,
    /// Exchanges X and Y axes.
    H_XY,
    /// Exchanges Y and Z axes.
    H_YZ,
    /// Negated X/Y-axis exchange.
    H_NXY,
    /// Negated X/Z-axis exchange.
    H_NXZ,
    /// Negated Y/Z-axis exchange.
    H_NYZ,
    // Pauli gates
    /// Identity.
    I,
    /// Pauli X.
    X,
    /// Pauli Y.
    Y,
    /// Pauli Z.
    Z,
    // Period 4 gates
    /// Positive square root of X.
    SQRT_X,
    /// Inverse square root of X.
    SQRT_X_DAG,
    /// Positive square root of Y.
    SQRT_Y,
    /// Inverse square root of Y.
    SQRT_Y_DAG,
    /// Positive square root of Z.
    #[strum(to_string = "S", serialize = "SQRT_Z")]
    S,
    /// Inverse square root of Z.
    #[strum(to_string = "S_DAG", serialize = "SQRT_Z_DAG")]
    S_DAG,
    // Period 3 gates (order-3 single-qubit Cliffords)
    /// Cycles X to Y to Z.
    C_XYZ,
    /// Cycles Z to Y to X.
    C_ZYX,
    /// Negated X/Y/Z cycle.
    C_NXYZ,
    /// X/negated-Y/Z cycle.
    C_XNYZ,
    /// X/Y/negated-Z cycle.
    C_XYNZ,
    /// Negated-X/Z/Y cycle.
    C_NZYX,
    /// Z/negated-Y/X cycle.
    C_ZNYX,
    /// Z/Y/negated-X cycle.
    C_ZYNX,
    // Non-Clifford Gate
    /// Positive pi/4 rotation about X.
    T_YZ,
    /// Negative pi/4 rotation about X.
    T_YZ_DAG,
    /// Positive pi/4 rotation about Y.
    T_XZ,
    /// Negative pi/4 rotation about Y.
    T_XZ_DAG,
    /// Positive pi/4 rotation about Z.
    #[strum(to_string = "T", serialize = "T_XY")]
    T,
    /// Negative pi/4 rotation about Z.
    #[strum(to_string = "T_DAG", serialize = "T_XY_DAG")]
    T_DAG,
}

impl GateType {
    /// Iterates over every gate. Lets a caller sweep the whole gate set —
    /// filtered by [`Self::is_reset`], [`Self::is_non_clifford`], and
    /// [`Self::is_two_qubit_gate`] — without depending on `strum` itself.
    pub fn iter() -> impl Iterator<Item = Self> {
        <Self as IntoEnumIterator>::iter()
    }

    /// Returns `true` for the reset gates ([`RX`](GateType::RX),
    /// [`RY`](GateType::RY), [`RZ`](GateType::RZ)).
    pub const fn is_reset(&self) -> bool {
        matches!(self, GateType::RX | GateType::RY | GateType::RZ)
    }

    /// Returns `true` for the non-Clifford [`T`](GateType::T)-family gates.
    pub const fn is_non_clifford(&self) -> bool {
        self.non_clifford_proxy().is_some()
    }

    /// Returns `true` for the two-qubit unitary (controlled) gates.
    pub const fn is_two_qubit_gate(&self) -> bool {
        self.two_qubit_bases().is_some()
    }

    /// Maps a non-Clifford gate to the Clifford that shares its axis and sign,
    /// so stabilizer-flow analysis can treat a [`T`](GateType::T) gate as its
    /// [`S`](GateType::S) proxy.
    /// Clifford gates map to themselves.
    pub const fn clifford_proxy(&self) -> GateType {
        match self.non_clifford_proxy() {
            Some(proxy) => proxy,
            None => *self,
        }
    }

    const fn non_clifford_proxy(&self) -> Option<GateType> {
        let proxy = match self {
            GateType::T => GateType::S,
            GateType::T_DAG => GateType::S_DAG,
            GateType::T_YZ => GateType::SQRT_X,
            GateType::T_YZ_DAG => GateType::SQRT_X_DAG,
            GateType::T_XZ => GateType::SQRT_Y,
            GateType::T_XZ_DAG => GateType::SQRT_Y_DAG,
            _ => return None,
        };
        Some(proxy)
    }

    /// For two-qubit gates in the `ACB` naming convention, returns the (control,
    /// target) Pauli bases — `CX` (= `ZCX`) is Z-control, X-target. `None` for
    /// non-two-qubit gates.
    pub const fn two_qubit_bases(&self) -> Option<(PauliBasis, PauliBasis)> {
        match self {
            GateType::CX => Some((PauliBasis::Z, PauliBasis::X)),
            GateType::CY => Some((PauliBasis::Z, PauliBasis::Y)),
            GateType::CZ => Some((PauliBasis::Z, PauliBasis::Z)),
            GateType::XCX => Some((PauliBasis::X, PauliBasis::X)),
            GateType::XCY => Some((PauliBasis::X, PauliBasis::Y)),
            GateType::XCZ => Some((PauliBasis::X, PauliBasis::Z)),
            GateType::YCX => Some((PauliBasis::Y, PauliBasis::X)),
            GateType::YCY => Some((PauliBasis::Y, PauliBasis::Y)),
            GateType::YCZ => Some((PauliBasis::Y, PauliBasis::Z)),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_mnemonics_match_display_and_parse() {
        for gate in GateType::iter() {
            let mnemonic: &'static str = gate.into();
            assert_eq!(mnemonic, gate.to_string());
            assert_eq!(mnemonic.parse::<GateType>().unwrap(), gate);
        }
    }
}
