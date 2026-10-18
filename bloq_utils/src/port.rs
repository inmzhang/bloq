use strum::{Display, EnumString};

/// Direction assigned to an open port.
///
/// Temporal ports may infer their role from time with [`Auto`](Self::Auto), or
/// choose [`Input`](Self::Input) or [`Output`](Self::Output) independently of time.
/// Spatial ports must choose an explicit role; [`Multiplex`](Self::Multiplex)
/// requires a spatial connection.
#[derive(
    Debug,
    Default,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    PartialOrd,
    Ord,
    serde::Serialize,
    serde::Deserialize,
    EnumString,
    Display,
)]
#[strum(ascii_case_insensitive)]
pub enum PortRole {
    /// Infer the role from temporal direction.
    #[default]
    Auto,
    /// Input boundary.
    Input,
    /// Output boundary.
    Output,
    /// Z-split boundary: input plus a correlated single-qubit output.
    Multiplex,
}

impl PortRole {
    /// Returns whether this role exposes an input boundary.
    #[must_use]
    pub const fn has_input_boundary(self) -> bool {
        matches!(self, Self::Input | Self::Multiplex)
    }

    /// Returns whether this role exposes an output boundary.
    #[must_use]
    pub const fn has_output_boundary(self) -> bool {
        matches!(self, Self::Output | Self::Multiplex)
    }

    /// Returns the lowercase serialized spelling.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Input => "input",
            Self::Output => "output",
            Self::Multiplex => "multiplex",
        }
    }
}
