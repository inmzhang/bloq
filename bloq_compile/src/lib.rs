//! Compile block graphs into physical circuit programs.
//!
//! Pipeline: [`BlockGraph`](bloq_graph::BlockGraph) → [`Bloq`] → backend emission.
//!
//! The compiler walks every block in a [`BlockGraph`](bloq_graph::BlockGraph), resolves its signature and
//! convention, generates per-block circuit templates, composes detectors and
//! observables from stabilizer flows, and lowers the result into Bloq
//! nodes suitable for backend emission.

mod block;
mod cache;
mod compile;
mod config;
mod error;
mod lower;
mod padding;
mod port;
mod signature;

pub(crate) type FxMap<K, V> = rustc_hash::FxHashMap<K, V>;
pub(crate) type FxSet<T> = rustc_hash::FxHashSet<T>;

/// A set usable as a GF(2) parity accumulator — see [`xor_toggle`].
pub(crate) trait XorToggleSet<T> {
    /// Insert `item` if absent, remove it if present.
    fn toggle(&mut self, item: T);
}

impl<T: Eq + std::hash::Hash, S: std::hash::BuildHasher> XorToggleSet<T>
    for std::collections::HashSet<T, S>
{
    fn toggle(&mut self, item: T) {
        if !self.remove(&item) {
            self.insert(item);
        }
    }
}

impl<T: Ord> XorToggleSet<T> for std::collections::BTreeSet<T> {
    fn toggle(&mut self, item: T) {
        if !self.remove(&item) {
            self.insert(item);
        }
    }
}

/// XOR a stream of items into `set`: an item seen an even number of times
/// cancels out of the parity. The one symmetric-difference kernel shared by
/// observable, gateway, and output-frame folding.
pub(crate) fn xor_toggle<T, S: XorToggleSet<T>>(set: &mut S, items: impl IntoIterator<Item = T>) {
    for item in items {
        set.toggle(item);
    }
}

/// XOR `items` into a parity multiset, returning the survivors sorted for
/// deterministic downstream ordering.
///
/// Sorting first and keeping one element per odd-length run is the same
/// symmetric difference as [`xor_toggle`] into a set, minus the hashing — and
/// the result had to be sorted regardless.
pub(crate) fn xor_toggled_sorted<T: Copy + Ord>(items: impl IntoIterator<Item = T>) -> Vec<T> {
    let mut all: Vec<T> = items.into_iter().collect();
    all.sort_unstable();
    all.chunk_by(|a, b| a == b)
        .filter(|run| run.len() % 2 == 1)
        .map(|run| run[0])
        .collect()
}

#[doc(no_inline)]
pub use bloq_ir::{
    Bloq, BloqEdge, BloqNode, BloqNodeId, NodeProvenance, SourceBlockRef, TemporalPipeRef,
};
pub use cache::SharedCompileCache;
pub(crate) use compile::BlockLayout;
pub use compile::{
    CLIFFORD_PROXY_SEED_METADATA_KEY, CODE_DISTANCE_METADATA_KEY, CONVENTION_METADATA_KEY,
    CompileArtifacts, CompileContext, CompileStage, CompiledObject, clear_clifford_proxy_cache,
    compile, compile_clifford_proxy, compile_detslice_proxy, compile_random_clifford_proxy,
    compile_with, compiled_distance,
};
pub use config::{CompileConfig, MAX_CODE_DISTANCE, spatial_hadamard_distance_warning};
pub use error::{CompileError, InvalidDistance, WalkError};
pub(crate) use error::{add_resource, check_resource};
pub use lower::validate_bloq_qubit_layout_for_source;
pub use signature::Connectivity;
