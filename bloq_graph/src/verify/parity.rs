//! Affine QuiZX parity evaluation and measurement keys.

use std::cmp::Ordering;

use glam::IVec3;
use quizx::params::{Parity, Var};

use crate::{Action, ActionNode, MeasureTarget, MeasurementObservable, Pauli, Stabilizer, ZXGraph};

pub(super) fn phase_constant(parity: &Parity) -> bool {
    // QuiZX exposes the variable iterator but no getter for the affine bit.
    parity != &Parity::new(parity.iter().collect::<Vec<_>>(), false)
}

pub(super) fn phase_value(parity: &Parity, mut value: impl FnMut(Var) -> bool) -> bool {
    parity.iter().fold(phase_constant(parity), |odd, variable| {
        odd ^ value(variable)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phase_evaluation_preserves_the_affine_bit() {
        for value in [false, true] {
            assert_eq!(phase_value(&Parity::new([7], false), |_| value), value);
            assert_eq!(phase_value(&Parity::new([7], true), |_| value), !value);
        }
        assert!(phase_value(&Parity::one(), |_| unreachable!()));
        assert!(!phase_value(&Parity::default(), |_| unreachable!()));
    }
}

/// Identifies a measurement record by the ZX site it reads and its Pauli basis.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) enum MeasurementKey {
    /// A single-node measurement at `pos` in the `pauli` basis.
    Node { pos: IVec3, pauli: Pauli },
    /// An edge measurement between `src` and `dst`; `pauli` is in the smaller
    /// node-ID endpoint's frame, so an H edge may use its flipped basis at `dst`.
    Edge {
        src: IVec3,
        dst: IVec3,
        pauli: Pauli,
    },
}

impl PartialOrd for MeasurementKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MeasurementKey {
    fn cmp(&self, other: &Self) -> Ordering {
        self.sort_key().cmp(&other.sort_key())
    }
}

impl MeasurementKey {
    pub(super) fn for_action(zx: &ZXGraph, action: &ActionNode) -> Option<Self> {
        let Action::Measure { target, .. } = action.action else {
            return None;
        };
        let pauli = match action.measurement? {
            MeasurementObservable::Concrete(basis) => Pauli::from(basis),
            MeasurementObservable::Selective(_) => Pauli::Y,
        };
        Some(match target {
            MeasureTarget::Node(pos) => Self::Node { pos, pauli },
            MeasureTarget::Edge { src, dir } => {
                let dst = src + dir.to_ivec3();
                let (src, dst) = if zx.node_at(src)?.id < zx.node_at(dst)?.id {
                    (src, dst)
                } else {
                    (dst, src)
                };
                Self::Edge { src, dst, pauli }
            }
        })
    }

    /// Original linear components whose OR is this measurement's physical support.
    pub(super) fn components(&self, zx: &ZXGraph) -> Option<Vec<(usize, Pauli)>> {
        let (column, pauli) = match *self {
            Self::Node { pos, pauli } => (zx.node_at(pos)?.id, pauli),
            Self::Edge { src, dst, pauli } => {
                let src = zx.node_at(src)?.id;
                let dst = zx.node_at(dst)?.id;
                (zx.edge_between(src, dst)?.id, pauli)
            }
        };
        Some(
            pauli
                .iter_xz()
                .flat_map(|axis| zx.readout_components(column, axis))
                .collect(),
        )
    }

    fn sort_key(&self) -> (u8, [i32; 3], [i32; 3], u8) {
        match *self {
            Self::Node { pos, pauli } => (0, pos.to_array(), [0; 3], u8::from(pauli)),
            Self::Edge { src, dst, pauli } => (1, src.to_array(), dst.to_array(), u8::from(pauli)),
        }
    }
}

/// Whether a stabilizer's interior support covers a measurement key — the
/// support predicate used by internal-measurement output correction.
pub(crate) fn stabilizer_supports_measurement(
    stabilizer: &Stabilizer,
    key: &MeasurementKey,
) -> bool {
    match key {
        MeasurementKey::Node { pos, pauli } => stabilizer
            .interior_nodes
            .get(pos)
            .is_some_and(|support| *support & *pauli),
        MeasurementKey::Edge { src, dst, pauli } => stabilizer
            .interior_edges
            .get(&(*src, *dst))
            .or_else(|| stabilizer.interior_edges.get(&(*dst, *src)))
            .is_some_and(|support| *support & *pauli),
    }
}
