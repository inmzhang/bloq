use std::borrow::Cow;
use std::collections::BTreeMap;
use std::fmt;
use std::ops::Index;

use bloq_circuit::PauliMap;
use bloq_graph::{Direction, Pauli};

use crate::signature::Connectivity;

const CENTER_BITS: u32 = 2;
const ARM_BITS: u32 = 3;
const ARM_MASK: u32 = 0b111;
const ARM_BASIS_MASK: u32 = 0b11;
const ARM_HADAMARD_BIT: u32 = 0b100;

/// A packed description of an observable's local flow through one block.
///
/// The low 2 bits store the junction Pauli. Each of the 6 directions then
/// uses 3 bits: 2 for the arm Pauli (`I/X/Z/Y`) and 1 for the Hadamard flag.
///
/// This is the physical flow shape consumed by an observable gateway, not the
/// coordinatewise restriction of a graph stabilizer row. Every Pauli component
/// carried by an arm also acts at the junction, so [`Self::with_arm`]
/// reconstructs that component in the center. Center-only components remain
/// possible for observables anchored or terminated inside the block.
///
/// This keeps the key compact while still allowing mixed local operators such
/// as an `X` arm on one face and a `Z` arm on another.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct LocalStabilizer(u32);

impl LocalStabilizer {
    /// Create a uniform-basis key with the given pipe subset.
    pub(crate) fn new(center_basis: Pauli, connectivity: Connectivity) -> Self {
        let mut local = Self::isolated(center_basis);
        for dir in Direction::iter() {
            if connectivity.has_pipe(dir) {
                local = local.with_arm(dir, center_basis, connectivity.has_hadamard(dir));
            }
        }
        local
    }

    /// Create a key with only a junction Pauli and no arm support.
    pub(crate) const fn isolated(center_basis: Pauli) -> Self {
        Self(pauli_bits(center_basis))
    }

    /// Physical junction Pauli at this block.
    pub(crate) fn center_basis(self) -> Pauli {
        decode_pauli_bits(self.0 & ARM_BASIS_MASK)
    }

    /// Pauli carried by the given arm, or `I` if the arm is inactive.
    pub(crate) fn arm_basis(self, dir: Direction) -> Pauli {
        decode_pauli_bits((self.0 >> arm_offset(dir)) & ARM_BASIS_MASK)
    }

    /// Whether the given direction carries a local operator arm.
    pub(crate) fn has_arm(self, dir: Direction) -> bool {
        self.arm_basis(dir) != Pauli::I
    }

    /// Whether the given arm corresponds to a Hadamard pipe.
    pub(crate) fn has_hadamard(self, dir: Direction) -> bool {
        self.has_arm(dir) && (self.0 >> arm_offset(dir)) & ARM_HADAMARD_BIT != 0
    }

    /// Add a previously inactive arm with the given basis and Hadamard flag.
    ///
    /// An arm is a logical flow into the block junction, so its Pauli
    /// components are also reconstructed in the center.
    pub(crate) fn with_arm(mut self, dir: Direction, basis: Pauli, hadamard: bool) -> Self {
        debug_assert!(!self.has_arm(dir), "a local flow arm is added only once");
        self.0 |= pauli_bits(basis);
        let shift = arm_offset(dir);
        self.0 &= !(ARM_MASK << shift);
        self.0 |= pauli_bits(basis) << shift;
        if basis != Pauli::I && hadamard {
            self.0 |= ARM_HADAMARD_BIT << shift;
        }
        self
    }

    /// Build the requested Pauli component of a local operator.
    ///
    /// The component exists if either the center node or any incident arm has
    /// support in the requested basis. `Pauli::I` has no support and returns
    /// `None` immediately.
    pub(crate) fn component(self, basis: Pauli) -> Option<Self> {
        if basis == Pauli::I {
            return None;
        }

        let has_center_support = self.center_basis() & basis;
        let mut has_support = has_center_support;
        let mut component = Self::isolated(if has_center_support { basis } else { Pauli::I });

        for dir in Direction::iter() {
            let arm_basis = self.arm_basis(dir);
            if arm_basis & basis {
                has_support = true;
                component = component.with_arm(dir, basis, self.has_hadamard(dir));
            }
        }

        has_support.then_some(component)
    }

    /// Iterate over active arm directions.
    pub(crate) fn arm_dirs(self) -> impl Iterator<Item = Direction> {
        Direction::iter().filter(move |dir| self.has_arm(*dir))
    }

    /// Connectivity induced by the active operator arms.
    pub(crate) fn connectivity(self) -> Connectivity {
        let mut connectivity = Connectivity::ISOLATED;
        for dir in Direction::iter() {
            if self.has_arm(dir) {
                connectivity = if self.has_hadamard(dir) {
                    connectivity.with_hadamard(dir)
                } else {
                    connectivity.with_pipe(dir)
                };
            }
        }
        connectivity
    }

    /// Shared arm Pauli when every active arm uses the same basis.
    pub(crate) fn uniform_arm_basis(self) -> Option<Pauli> {
        let mut basis = None;
        for dir in self.arm_dirs() {
            let arm_basis = self.arm_basis(dir);
            match basis {
                Some(existing) if existing != arm_basis => return None,
                Some(_) => {}
                None => basis = Some(arm_basis),
            }
        }
        basis
    }

    /// Number of active pipe directions.
    pub(crate) fn weight(self) -> u32 {
        self.arm_dirs().count() as u32
    }

    /// Whether no pipes carry the operator (weight == 0).
    pub(crate) fn is_isolated(self) -> bool {
        self.weight() == 0
    }
}

impl fmt::Debug for LocalStabilizer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_isolated() {
            return write!(f, "LocalStabilizer({}, ISOLATED)", self.center_basis());
        }

        if self.uniform_arm_basis() == Some(self.center_basis()) {
            write!(
                f,
                "LocalStabilizer({}, {:?})",
                self.center_basis(),
                self.connectivity(),
            )?;
            return Ok(());
        }

        write!(f, "LocalStabilizer({}, Arms(", self.center_basis())?;
        let mut first = true;
        for dir in Direction::iter() {
            let arm_basis = self.arm_basis(dir);
            if arm_basis == Pauli::I {
                continue;
            }
            if !first {
                write!(f, " ")?;
            }
            first = false;
            if self.has_hadamard(dir) {
                write!(f, "{dir}:{arm_basis}H")?;
            } else {
                write!(f, "{dir}:{arm_basis}")?;
            }
        }
        write!(f, "))")
    }
}

impl fmt::Display for LocalStabilizer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

impl From<LocalStabilizer> for u32 {
    fn from(local_stabilizer: LocalStabilizer) -> Self {
        local_stabilizer.0
    }
}

/// Gateway-key connectivity for a chosen set of a block's pipe directions.
///
/// Keys never carry Hadamard flags: the pipe owning the X↔Z flip (a realignment
/// node, or a spatial Hadamard wall) answers it in its own gateway, arm Paulis
/// are already frame-local (`incident_edge_pauli` flips the stored Pauli when
/// read across a Hadamard edge), and observable lowering builds every query key
/// flag-free — so a flagged key would be unreachable by construction.
pub(crate) fn gateway_key_connectivity(dirs: impl IntoIterator<Item = Direction>) -> Connectivity {
    dirs.into_iter()
        .fold(Connectivity::ISOLATED, Connectivity::with_pipe)
}

/// Measurements in a specific chunk contributing to an observable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ChunkMeasurements {
    pub(crate) chunk_index: usize,
    pub(crate) measurements: Vec<u32>,
}

/// Record parity and support-only boundary operators for one local flow.
///
/// Operators use block-local coordinates at the −Z and +Z temporal faces;
/// an empty map means the flow does not cross that face. Lowering translates
/// operator support and binds template records to instance measurements.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(crate) struct GatewayEntry {
    pub(crate) measurements: Vec<ChunkMeasurements>,
    pub(crate) operator_in: PauliMap,
    pub(crate) operator_out: PauliMap,
}

impl GatewayEntry {
    /// XOR-combine multiple exact gateway entries into one mixed observable entry.
    ///
    /// Measurements combine by symmetric difference (a record appearing an even
    /// number of times cancels). The boundary operators combine by the pointwise
    /// Pauli product (`PauliMap` `Mul`), so decomposing a `Y` request into its
    /// `X` and `Z` components and XOR-ing them reconstructs the `Y = X·Z`
    /// boundary operator on each face.
    pub(crate) fn xor<'a>(entries: impl IntoIterator<Item = &'a GatewayEntry>) -> Self {
        let mut by_chunk: BTreeMap<usize, Vec<u32>> = BTreeMap::new();
        let mut operator_in = PauliMap::empty();
        let mut operator_out = PauliMap::empty();
        for entry in entries {
            for chunk_measurements in &entry.measurements {
                by_chunk
                    .entry(chunk_measurements.chunk_index)
                    .or_default()
                    .extend(&chunk_measurements.measurements);
            }
            operator_in = &operator_in ^ &entry.operator_in;
            operator_out = &operator_out ^ &entry.operator_out;
        }

        let measurements = by_chunk
            .into_iter()
            .filter_map(|(chunk_index, bucket)| {
                let measurements = crate::xor_toggled_sorted(bucket);
                (!measurements.is_empty()).then_some(ChunkMeasurements {
                    chunk_index,
                    measurements,
                })
            })
            .collect();

        Self {
            measurements,
            operator_in,
            operator_out,
        }
    }
}

/// Maps operator routing -> measurements. Missing key = forbidden configuration.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(crate) struct ObservableGateway(crate::FxMap<LocalStabilizer, GatewayEntry>);

impl ObservableGateway {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn insert(
        &mut self,
        key: LocalStabilizer,
        entry: GatewayEntry,
    ) -> Option<GatewayEntry> {
        self.0.insert(key, entry)
    }

    /// Bind the first-round alias of a joint check to each surface using one
    /// of its incident wall arms. A surface turning through both arms owns it once.
    pub(crate) fn bind_wall_corner(&mut self, arms: &[(Direction, Pauli)], id: u32) {
        for (key, entry) in &mut self.0 {
            if arms
                .iter()
                .any(|&(direction, basis)| key.arm_basis(direction) & basis)
            {
                entry.measurements.push(ChunkMeasurements {
                    chunk_index: 0,
                    measurements: vec![id],
                });
            }
        }
    }

    /// Rewrite every measurement id from chunk-local space into template space.
    ///
    /// Block compilers build the gateway while measurements are still numbered
    /// per chunk; [`LoweringTemplate::from_chunks`] calls this once the template
    /// numbering is assigned, so observable lowering maps gateway measurements
    /// straight to instance measurements without re-deriving the chunk-local
    /// translation. Each `ChunkMeasurements::chunk_index` is retained as the
    /// originating chunk, but its `measurements` are sorted, XOR-canonical
    /// template ids afterward so repeated lowering can reuse the range directly.
    pub(crate) fn resolve_to_template_ids(&mut self, mut resolve: impl FnMut(usize, u32) -> u32) {
        for entry in self.0.values_mut() {
            for chunk in &mut entry.measurements {
                for measurement in &mut chunk.measurements {
                    *measurement = resolve(chunk.chunk_index, *measurement);
                }
                chunk.measurements =
                    crate::xor_toggled_sorted(std::mem::take(&mut chunk.measurements));
            }
        }
    }

    /// Shift every entry's originating chunk index, for harnesses that
    /// prepend chunks (e.g. a perfect-initialization port) before handing the
    /// gateway to `LoweringTemplate::from_chunks`.
    #[cfg(test)]
    pub(crate) fn shift_chunk_indices(&mut self, offset: usize) {
        for entry in self.0.values_mut() {
            for chunk in &mut entry.measurements {
                chunk.chunk_index += offset;
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn contains_key(&self, key: &LocalStabilizer) -> bool {
        self.0.contains_key(key)
    }

    #[cfg(test)]
    pub(crate) fn keys(&self) -> impl Iterator<Item = &LocalStabilizer> {
        self.0.keys()
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.0.len()
    }

    /// Resolve one local observable request.
    ///
    /// Current behavior is intentionally simple:
    /// 1. exact-key lookup
    /// 2. fallback decomposition into global X/Z components and XOR-combine
    ///
    /// This is not a general GF(2) basis solver over arbitrary gateway keys.
    pub(crate) fn lookup(&self, local_stabilizer: LocalStabilizer) -> Option<GatewayEntry> {
        self.lookup_ref(local_stabilizer).map(Cow::into_owned)
    }

    pub(crate) fn lookup_ref(
        &self,
        local_stabilizer: LocalStabilizer,
    ) -> Option<Cow<'_, GatewayEntry>> {
        if let Some(entry) = self.0.get(&local_stabilizer) {
            return Some(Cow::Borrowed(entry));
        }

        let entries = [Pauli::X, Pauli::Z]
            .into_iter()
            .filter_map(|basis| local_stabilizer.component(basis))
            .map(|component| self.0.get(&component))
            .collect::<Option<Vec<_>>>()?;
        (!entries.is_empty()).then(|| Cow::Owned(GatewayEntry::xor(entries)))
    }
}

impl Index<&LocalStabilizer> for ObservableGateway {
    type Output = GatewayEntry;

    fn index(&self, index: &LocalStabilizer) -> &Self::Output {
        &self.0[index]
    }
}

const fn pauli_bits(pauli: Pauli) -> u32 {
    pauli as u32
}

fn decode_pauli_bits(bits: u32) -> Pauli {
    match bits {
        0 => Pauli::I,
        1 => Pauli::X,
        2 => Pauli::Z,
        3 => Pauli::Y,
        _ => unreachable!("Pauli is encoded in two bits"),
    }
}

fn arm_offset(dir: Direction) -> u32 {
    CENTER_BITS + dir.index() as u32 * ARM_BITS
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gateway_entry(qubit_x: i32) -> GatewayEntry {
        GatewayEntry {
            measurements: vec![ChunkMeasurements {
                chunk_index: 0,
                measurements: vec![qubit_x as u32],
            }],
            ..Default::default()
        }
    }

    #[test]
    fn test_with_pipes() {
        let conn = Connectivity::ISOLATED
            .with_pipe(Direction::XPLUS)
            .with_pipe(Direction::YMINUS);
        let ls = LocalStabilizer::new(Pauli::Z, conn);
        assert!(!ls.is_isolated());
        assert_eq!(ls.weight(), 2);
        assert_eq!(ls.center_basis(), Pauli::Z);
        assert_eq!(ls.arm_basis(Direction::XPLUS), Pauli::Z);
        assert_eq!(ls.arm_basis(Direction::YMINUS), Pauli::Z);
    }

    #[test]
    fn test_mixed_arms() {
        let ls = LocalStabilizer::isolated(Pauli::Y)
            .with_arm(Direction::XPLUS, Pauli::X, false)
            .with_arm(Direction::ZPLUS, Pauli::Z, true);
        assert_eq!(ls.arm_basis(Direction::XPLUS), Pauli::X);
        assert_eq!(ls.arm_basis(Direction::ZPLUS), Pauli::Z);
        assert!(ls.has_hadamard(Direction::ZPLUS));
        assert_eq!(format!("{ls:?}"), "LocalStabilizer(Y, Arms(+X:X +Z:ZH))");
    }

    #[test]
    fn test_component_projection() {
        let ls = LocalStabilizer::isolated(Pauli::Y)
            .with_arm(Direction::XPLUS, Pauli::Y, false)
            .with_arm(Direction::XMINUS, Pauli::X, false);
        let x_component = ls.component(Pauli::X).expect("X component");
        let z_component = ls.component(Pauli::Z).expect("Z component");
        assert_eq!(
            x_component,
            LocalStabilizer::isolated(Pauli::X)
                .with_arm(Direction::XPLUS, Pauli::X, false)
                .with_arm(Direction::XMINUS, Pauli::X, false)
        );
        assert_eq!(
            z_component,
            LocalStabilizer::isolated(Pauli::Z).with_arm(Direction::XPLUS, Pauli::Z, false)
        );
    }

    #[test]
    fn test_arm_flow_reconstructs_the_center() {
        let flow = LocalStabilizer::isolated(Pauli::I)
            .with_arm(Direction::XMINUS, Pauli::X, false)
            .with_arm(Direction::YPLUS, Pauli::X, false);

        assert_eq!(flow.center_basis(), Pauli::X);
        assert_eq!(
            flow,
            LocalStabilizer::new(
                Pauli::X,
                Connectivity::ISOLATED
                    .with_pipe(Direction::XMINUS)
                    .with_pipe(Direction::YPLUS),
            ),
        );
    }

    #[test]
    fn test_component_handles_y_and_i() {
        let ls = LocalStabilizer::isolated(Pauli::Y).with_arm(Direction::XPLUS, Pauli::Y, false);
        assert_eq!(ls.component(Pauli::Y), Some(ls));
        assert_eq!(ls.component(Pauli::I), None);
    }

    #[test]
    fn test_lookup_prefers_exact_hit_before_decomposition_and_otherwise_decomposes() {
        let conn = Connectivity::ISOLATED.with_pipe(Direction::XPLUS);
        let x_key = LocalStabilizer::new(Pauli::X, conn);
        let z_key = LocalStabilizer::new(Pauli::Z, conn);
        let y_key = LocalStabilizer::new(Pauli::Y, conn);

        let x_entry = gateway_entry(0);
        let z_entry = gateway_entry(1);
        let exact_y = gateway_entry(2);
        let mut exact_gateway = ObservableGateway::new();
        exact_gateway.insert(y_key, exact_y.clone());
        exact_gateway.insert(x_key, x_entry.clone());
        exact_gateway.insert(z_key, z_entry.clone());

        assert_eq!(exact_gateway.lookup(y_key), Some(exact_y));

        let mut decomposed_gateway = ObservableGateway::new();
        decomposed_gateway.insert(x_key, x_entry.clone());
        decomposed_gateway.insert(z_key, z_entry.clone());

        assert_eq!(
            decomposed_gateway.lookup(y_key),
            Some(GatewayEntry {
                measurements: vec![ChunkMeasurements {
                    chunk_index: 0,
                    measurements: vec![0, 1],
                }],
                ..Default::default()
            })
        );
        assert!(
            decomposed_gateway
                .lookup(LocalStabilizer::isolated(Pauli::I))
                .is_none()
        );

        let mut incomplete_gateway = ObservableGateway::new();
        incomplete_gateway.insert(x_key, x_entry);
        assert!(incomplete_gateway.lookup(y_key).is_none());
    }

    #[test]
    fn test_lookup_decomposes_reconstructed_flow_components() {
        let local = LocalStabilizer::isolated(Pauli::X).with_arm(Direction::XPLUS, Pauli::Z, false);
        let x_entry = gateway_entry(0);
        let z_entry = gateway_entry(1);
        let x_component = LocalStabilizer::isolated(Pauli::X);
        let z_component =
            LocalStabilizer::isolated(Pauli::Z).with_arm(Direction::XPLUS, Pauli::Z, false);

        let mut gateway = ObservableGateway::new();
        gateway.insert(x_component, x_entry.clone());
        gateway.insert(z_component, z_entry.clone());
        assert_eq!(
            gateway.lookup(local),
            Some(GatewayEntry {
                measurements: vec![ChunkMeasurements {
                    chunk_index: 0,
                    measurements: vec![0, 1],
                }],
                ..Default::default()
            })
        );
    }

    #[test]
    fn resolving_measurements_canonicalizes_each_chunk() {
        let key = LocalStabilizer::isolated(Pauli::X);
        let mut gateway = ObservableGateway::new();
        gateway.insert(
            key,
            GatewayEntry {
                measurements: vec![ChunkMeasurements {
                    chunk_index: 3,
                    measurements: vec![5, 2, 1, 2, 1],
                }],
                ..Default::default()
            },
        );

        gateway.resolve_to_template_ids(|_, measurement| measurement);

        assert_eq!(gateway[&key].measurements[0].measurements, [5]);
    }
}
