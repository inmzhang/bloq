use bloq_graph::{BlockGraph, ModuleCertificationLimits};

use crate::error::InvalidDistance;

/// Largest supported surface code distance.
///
/// Fixed-bulk lowering materializes grids with area quadratic in the distance;
/// this practical limit rejects inputs that would otherwise exhaust memory.
pub const MAX_CODE_DISTANCE: u32 = 255;

/// Parameters that determine how a [`BlockGraph`](bloq_graph::BlockGraph) is
/// lowered: the surface code distance and T-source mode.
///
/// A valid surface code distance is odd and in `3..=MAX_CODE_DISTANCE`; the
/// upper bound limits the compiler's quadratic grid materialization. The fields
/// are private so this invariant holds by construction. Build one with
/// [`try_new`] (the fallible public path) or [`new`] (panics on an invalid
/// distance, for known-good internal literals).
///
/// [`try_new`]: CompileConfig::try_new
/// [`new`]: CompileConfig::new
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CompileConfig {
    code_distance: u32,
    prepare_t_with_mpps: bool,
    certification_limits: ModuleCertificationLimits,
}

impl CompileConfig {
    /// Whether `code_distance` is a distance the compiler accepts: odd and in
    /// `3..=`[`MAX_CODE_DISTANCE`].
    ///
    /// The predicate form of [`try_new`](Self::try_new), for front ends that
    /// validate an input field before there is a graph to compile.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloq_compile::CompileConfig;
    ///
    /// assert!(CompileConfig::is_valid_distance(3));
    /// assert!(!CompileConfig::is_valid_distance(4));
    /// assert!(!CompileConfig::is_valid_distance(1));
    /// ```
    #[must_use]
    pub const fn is_valid_distance(code_distance: u32) -> bool {
        code_distance >= 3 && code_distance <= MAX_CODE_DISTANCE && code_distance % 2 == 1
    }

    /// Builds a config, validating the documented distance range and parity.
    ///
    /// # Errors
    ///
    /// Returns [`InvalidDistance`] unless
    /// [`is_valid_distance`](Self::is_valid_distance) accepts `code_distance`.
    pub fn try_new(code_distance: u32) -> Result<Self, InvalidDistance> {
        if !Self::is_valid_distance(code_distance) {
            return Err(InvalidDistance(code_distance));
        }
        Ok(Self {
            code_distance,
            prepare_t_with_mpps: false,
            certification_limits: ModuleCertificationLimits::DEFAULT,
        })
    }

    /// Builds a config from a known-valid distance.
    ///
    /// # Panics
    ///
    /// Panics if `code_distance` is not odd and in `3..=255`; use [`try_new`](Self::try_new)
    /// for distances that come from user input.
    #[must_use]
    pub fn new(code_distance: u32) -> Self {
        Self::try_new(code_distance).expect("code distance must be odd and in 3..=255")
    }

    /// The surface code distance (odd and in `3..=MAX_CODE_DISTANCE`).
    #[must_use]
    pub fn code_distance(&self) -> u32 {
        self.code_distance
    }

    /// Whether T blocks use the compact product-state preparation followed by
    /// one stabilizer `MPP`, instead of MSC-LS cultivation.
    #[must_use]
    pub fn prepare_t_with_mpps(&self) -> bool {
        self.prepare_t_with_mpps
    }

    /// Returns this config with compact T-block preparation enabled or disabled.
    #[must_use]
    pub const fn with_prepare_t_with_mpps(mut self, enabled: bool) -> Self {
        self.prepare_t_with_mpps = enabled;
        self
    }

    /// Limits applied while certifying a module hierarchy.
    #[must_use]
    pub const fn certification_limits(&self) -> ModuleCertificationLimits {
        self.certification_limits
    }

    /// Returns this config with explicit module-certification limits.
    #[must_use]
    pub const fn with_certification_limits(mut self, limits: ModuleCertificationLimits) -> Self {
        self.certification_limits = limits;
        self
    }
}

/// Compatibility entry point for callers checking the former spatial-wall advisory.
/// Flagged spatial Hadamard extraction no longer emits that advisory.
#[must_use]
pub fn spatial_hadamard_distance_warning(_graph: &BlockGraph) -> Option<&'static str> {
    None
}

impl Default for CompileConfig {
    fn default() -> Self {
        Self::new(3)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distance_bound_limits_quadratic_grid_materialization() {
        assert_eq!(
            CompileConfig::try_new(MAX_CODE_DISTANCE)
                .unwrap()
                .code_distance(),
            MAX_CODE_DISTANCE
        );
        assert_eq!(
            CompileConfig::try_new(MAX_CODE_DISTANCE + 2),
            Err(InvalidDistance(MAX_CODE_DISTANCE + 2))
        );
    }
}
