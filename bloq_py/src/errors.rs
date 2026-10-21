//! Python domain exceptions share `BloqError`; Rust variants appear in their
//! messages. `InvalidArgumentError` handles invalid domain arguments.
//!
//! Domain failures derive from `BloqError`. Sequence and mapping operations
//! keep `IndexError` and `KeyError`; argument coercions may raise `TypeError`
//! or `OverflowError`, and filesystem access raises `OSError`.
//!
//! `CompileWarning` derives from `UserWarning`, separately from the errors.

use std::path::PathBuf;

use pyo3::exceptions::{PyException, PyOSError, PyUserWarning};
use pyo3::prelude::*;

/// Invalid domain arguments use the shared `BloqError` hierarchy.
pub(crate) fn invalid_argument(e: impl std::fmt::Display) -> PyErr {
    InvalidArgumentError::new_err(e.to_string())
}

/// Preserve the path and OS code; Python selects the matching `OSError` subclass.
pub(crate) fn io_error(path: PathBuf, source: &std::io::Error) -> PyErr {
    let Some(errno) = source.raw_os_error() else {
        return std::io::Error::new(source.kind(), format!("{}: {source}", path.display())).into();
    };
    #[cfg(not(windows))]
    let args = (errno, source.to_string(), path.into_os_string());
    #[cfg(windows)]
    let args = (0, source.to_string(), path.into_os_string(), errno);
    PyOSError::new_err(args)
}

/// Declares the whole exception hierarchy AND generates `register()` from the
/// same list, so adding an exception is a single edit — a declaration that is
/// missing its registration cannot happen by construction.
///
/// Per item this works like `pyo3_stub_gen::create_exception!`, but references
/// the exception by its unqualified name in stubs. The upstream macro uses
/// `TypeInfo::builtin`, which renders subclass bases as e.g.
/// `builtins.BloqError` — an invalid annotation for a locally defined class.
macro_rules! bloq_exceptions {
    ($($name:ident : $base:ty => $doc:expr;)+) => {
        $(bloq_exceptions!(@one $name, $base, $doc);)+

        pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
            let py = m.py();
            $(m.add(stringify!($name), py.get_type::<$name>())?;)+
            Ok(())
        }
    };
    (@one $name:ident, $base:ty, $doc:expr) => {
        pyo3::create_exception!(bloq._core, $name, $base, $doc);

        impl pyo3_stub_gen::PyStubType for $name {
            fn type_output() -> pyo3_stub_gen::TypeInfo {
                pyo3_stub_gen::TypeInfo::unqualified(stringify!($name))
            }
        }

        pyo3_stub_gen::impl_py_runtime_type!($name);

        pyo3_stub_gen::inventory::submit! {
            pyo3_stub_gen::type_info::PyClassInfo {
                pyclass_name: stringify!($name),
                struct_id: std::any::TypeId::of::<$name>,
                getters: &[],
                setters: &[],
                module: Some("bloq._core"),
                doc: $doc,
                bases: &[|| <$base as pyo3_stub_gen::PyStubType>::type_output()],
                has_eq: false,
                has_ord: false,
                has_hash: false,
                has_str: false,
                subclass: true,
            }
        }
    };
}

bloq_exceptions! {
    BloqError: PyException =>
"Base exception for bloq domain failures.

Python protocol, argument-conversion, and I/O errors retain their standard
builtins, including IndexError, KeyError, TypeError, OverflowError, and OSError.

Examples:
    >>> import bloq
    >>> issubclass(bloq.CompileError, bloq.BloqError)
    True
    >>> issubclass(bloq.ParseError, bloq.BloqError)
    True
";
    InvalidArgumentError: BloqError =>
        "An argument failed validation, or a lookup key (id, name, position, \
         basis/kind spelling) was not found.";
    ParseError: BloqError =>
        "A `.blog` source failed to parse (message carries the rendered diagnostic).";
    BlockGraphError: BloqError =>
        "A block-graph operation or validation failed.";
    CompileError: BloqError =>
"Lowering a block graph to Bloq IR failed.

The ``layout_conflict`` attribute is None except for physical ownership
conflicts. Its dictionary contains ``kind``, the physical ``qubit`` coordinate,
and conflicting nodes: ``producer``, ``consumer``, ``reuser`` for live-state
reuse, or ``first``, ``second`` for same-layer overlap. Each node contains its
graph-local ``id`` and sorted source ``blocks`` coordinates. Pipe endpoints
and nested region members are included; synthetic nodes may have no blocks.
``related`` contains additional reuse witnesses from the same frozen failed
scheduling state. This bounded diagnostic list may be partial or empty.
";
    LowerError: BloqError =>
        "Lowering Bloq IR to a VM program failed.";
    BloqValidationError: BloqError =>
        "A Bloq IR program failed validation.";
    TextParseError: BloqError =>
        "A `.bloqir` text payload failed to parse.";
    BinaryDecodeError: BloqError =>
        "A `.bloq` binary payload failed to decode.";
    StimEmissionError: BloqError =>
        "Emitting a Stim circuit from Bloq IR failed.";
    RuntimeError: BloqError =>
        "Executing a lowered VM program failed.";
    CompileWarning: PyUserWarning =>
"An advisory raised by a successful compile.

A `UserWarning` subclass, so it is filterable on its own:
`warnings.filterwarnings(\"error\", category=bloq.CompileWarning)` turns
compile advisories into exceptions without touching other warnings.

Examples:
    >>> import bloq, warnings
    >>> issubclass(bloq.CompileWarning, UserWarning)
    True
";
}
