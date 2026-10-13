//! Shared utility types for the bloq quantum compiler.
//!
//! Provides foundational types used across the workspace: Pauli operators
//! ([`Pauli`], [`PauliBasis`], [`PauliString`], [`PhasedPauliString`]), basis types ([`Basis`]),
//! spatial directions ([`Direction`], [`UDirection`]), and block-graph colors
//! ([`RGBA`]).
//!
//! This crate owns no quantum engine. Graph verification uses QuiZX, while
//! compiled-circuit execution uses `ticit`; both stay in their owning crates.

pub mod boolean;
mod color;
mod dir;
pub mod graph_layout;
mod pauli;
mod port;

pub use color::RGBA;
pub use dir::{Direction, DirectionParseError, UDirection};
pub use pauli::{Basis, Pauli, PauliBasis, PauliError, PauliString, PhasedPauliString};
pub use port::PortRole;
