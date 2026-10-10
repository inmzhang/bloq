//! Compile block graphs for fault-tolerant quantum circuits.
//!
//! Workspace crates are available as [`graph`], [`compile`], [`ir`], and other
//! modules. Import common types and entry points with [`prelude`].
//! Use [`Result`] for workflows spanning these modules; [`Error`] preserves
//! their typed failures and supports automatic conversion with `?`.
//!
//! The whole pipeline in one call:
//!
//! ```
//! use bloq::prelude::*;
//!
//! let stim_text = compile_to_stim(&GalleryItem::CNOT.build(), 3)?;
//! assert!(stim_text.contains("QUBIT_COORDS"));
//! # Ok::<(), bloq::Error>(())
//! ```
//!
//! Spelled out, when the intermediate [`Bloq`](bloq_ir::Bloq) is wanted or a
//! non-default [`CompileConfig`](bloq_compile::CompileConfig) applies:
//!
//! ```
//! use bloq::prelude::*;
//!
//! let graph = GalleryItem::CNOT.build();
//! let bloq = compile(&graph, 3)?;
//! let stim_text = emit_bloq_stim(&bloq)?;
//! assert!(stim_text.contains("QUBIT_COORDS"));
//! # Ok::<(), bloq::Error>(())
//! ```
//!
//! Compiling many related graphs with one configuration can use a
//! [`CompileContext`](bloq_compile::CompileContext) to reuse their circuit
//! templates.
//!
//! Default features are empty, so library use pulls in no CLI dependencies.
//! The separate `bloq-cli` package provides the `bloq` command-line binary
//! and the shared runner used by Python. `vm` exposes physical execution. `gltf` and
//! `verify` forward to
//! `bloq_graph/gltf` and `bloq_stim/verify` respectively. `graph-verify` exposes
//! graph-level logical verification without requiring the native Stim backend.

pub use bloq_circuit as circuit;
pub use bloq_compile as compile;
pub use bloq_graph as graph;
pub use bloq_ir as ir;
pub use bloq_stim as stim;
pub use bloq_utils as utils;
#[cfg(feature = "vm")]
pub use bloq_vm as vm;

mod error;
pub use error::{Error, Result};

use bloq_graph::BlockGraph;

/// Compile `graph` at `distance` and emit it as Stim circuit text.
///
/// Uses a fresh template cache. Equivalent to
/// [`bloq_compile::compile`] followed by [`bloq_stim::emit_bloq_stim`].
///
/// # Errors
///
/// Wraps [`compile::CompileError`] if `distance` is invalid or `graph` does not
/// compile, and [`stim::StimEmissionError`] if the compiled program uses a
/// node the static Stim backend cannot express (dynamic `RepeatUntilSuccess` or
/// retry regions or unpinned quantum membership — use `bloq_vm` for dynamic execution).
///
/// # Examples
///
/// ```
/// let stim_text = bloq::compile_to_stim(&bloq::graph::GalleryItem::CNOT.build(), 3)?;
/// assert!(stim_text.contains("QUBIT_COORDS"));
/// # Ok::<(), bloq::Error>(())
/// ```
pub fn compile_to_stim(graph: &BlockGraph, distance: u32) -> Result<String> {
    let bloq = bloq_compile::compile(graph, distance)?;
    Ok(bloq_stim::emit_bloq_stim(&bloq)?)
}

/// Graph authoring, compilation, and Stim emission, for `use bloq::prelude::*`.
///
/// Advanced compiler controls live in [`crate::compile`]; circuit construction
/// and physical IR editing use [`crate::circuit`] and [`crate::ir`], including
/// their respective preludes.
///
/// Load and compile a module hierarchy from disk:
///
/// ```no_run
/// use bloq::prelude::*;
///
/// let graph = BlockGraph::load("input.blog")?;
/// let config = CompileConfig::try_new(5)?;
/// let artifacts = CompileContext::new(config).compile(&graph)?;
/// std::fs::write("out.stim", emit_bloq_stim(&artifacts.bloq)?)?;
/// # Ok::<(), bloq::Error>(())
/// ```
pub mod prelude {
    #[doc(no_inline)]
    pub use bloq_circuit::NoiseModel;
    #[doc(no_inline)]
    pub use bloq_compile::{
        CompileArtifacts, CompileConfig, CompileContext, CompileError, compile, compile_with,
    };
    #[doc(no_inline)]
    pub use bloq_graph::prelude::*;
    #[doc(no_inline)]
    pub use bloq_ir::Bloq;
    #[doc(no_inline)]
    pub use bloq_stim::prelude::*;

    #[doc(no_inline)]
    pub use crate::{Error, Result, compile_to_stim};

    /// Integer coordinates for block graphs and qubit layouts.
    #[doc(no_inline)]
    pub use glam::IVec3;
}

// Compile the user documentation in the existing doctest gate without adding
// its prose to the facade's published API reference.
#[cfg(doctest)]
mod documentation {
    #[doc = include_str!("../../README.md")]
    mod readme {}
    #[doc = include_str!("../../docs/guide.md")]
    mod guide {}
    #[doc = include_str!("../../docs/releasing.md")]
    mod releasing {}
}

#[cfg(test)]
mod tests {
    use super::*;

    fn module_stim(source: &str, distance: u32) -> Result<String> {
        use crate::prelude::*;

        let source = BlockGraph::from_text(source)?;
        let config = CompileConfig::try_new(distance)?;
        let artifacts = CompileContext::new(config).compile(&source)?;
        let program = Bloq::from_binary(&artifacts.bloq.to_binary())?;
        let program = Bloq::from_text(&program.to_text())?;
        program.validate()?;
        Ok(emit_bloq_stim(&program)?)
    }

    #[test]
    fn facade_result_composes_modules_compilation_and_codecs() -> Result {
        use std::error::Error as _;

        let source = graph::GalleryItem::CNOT.build().to_blog_text();
        assert!(module_stim(&source, 3)?.contains("QUBIT_COORDS"));
        let error = module_stim("invalid BLOG", 3).unwrap_err();
        assert!(error.is::<graph::BlockGraphError>());
        assert!(error.source().unwrap().is::<graph::BlockGraphError>());
        assert!(matches!(
            module_stim(&source, 4).unwrap_err().downcast_ref(),
            Some(compile::InvalidDistance(4))
        ));
        Ok(())
    }

    #[test]
    fn facade_error_retains_typed_sources_and_resource_failures() {
        use std::error::Error as _;

        fn assert_thread_safe<E: std::error::Error + Send + Sync + 'static>() {}
        assert_thread_safe::<Error>();

        let resource = utils::boolean::BooleanResourceError {
            resource: "Boolean steps",
            observed: 2,
            limit: 1,
        };
        let source = compile::CompileError::from(resource);
        let message = source.to_string();
        let error = Error::from(source);
        assert_eq!(error.to_string(), message);
        let source = error
            .source()
            .unwrap()
            .downcast_ref::<compile::CompileError>()
            .unwrap();
        assert!(source.is_resource_limited());
        assert!(
            matches!(source, compile::CompileError::BooleanResource(actual) if *actual == resource)
        );
        assert!(error.is::<compile::CompileError>());
        assert!(error.downcast_ref::<graph::ParseError>().is_none());

        let error = Error::from(std::io::Error::other("output write failed"));
        assert_eq!(error.to_string(), "output write failed");
        assert!(error.source().unwrap().is::<std::io::Error>());
    }

    #[cfg(feature = "vm")]
    #[test]
    fn facade_result_drives_vm_and_serializes_artifacts() -> Result {
        let program = compile::compile(&graph::GalleryItem::CNOT.build(), 3)?;
        let config = vm::LoweringConfig::default();
        let program = vm::lower(&program, &config)?;
        let result = program.run(config.runtime_config(42))?;
        assert!(!result.artifact.discarded);
        assert!(!result.artifact.to_json()?.is_empty());
        Ok(())
    }

    /// The two failure domains stay distinguishable: a bad distance is a
    /// compile error, and a dynamic program is an emission error.
    #[test]
    fn compile_to_stim_reports_which_stage_failed() {
        assert!(matches!(
            compile_to_stim(&graph::GalleryItem::CNOT.build(), 4)
                .unwrap_err()
                .downcast_ref(),
            Some(compile::CompileError::InvalidDistance(4))
        ));
        assert!(
            compile_to_stim(&graph::GalleryItem::T.build(), 3)
                .unwrap_err()
                .is::<stim::StimEmissionError>()
        );
    }
}
