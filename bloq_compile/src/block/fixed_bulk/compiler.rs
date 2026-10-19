use bloq_graph::{BlockKind, Direction};

use crate::CompileError;
use crate::block::CompiledTemplate;
use crate::signature::BlockSignature;

use crate::block::walk::compile_walking;

use super::compile_cube;
use super::compile_measurement;
use super::port::compile_port;
use super::rotation::compile_patch_rotation;
use super::ybasis::compile_y;

pub(crate) fn validate_fixed_bulk(signature: BlockSignature) -> Result<(), CompileError> {
    let connectivity = signature.connectivity;
    let missing_basis = signature.boundary_basis.is_none();
    // Preserve Y's historical diagnostic precedence when both fields are invalid.
    if signature.kind == BlockKind::Y && missing_basis {
        return Err(CompileError::MissingBoundaryBasis {
            kind: signature.kind,
        });
    }
    let past = connectivity.has_pipe(Direction::ZMINUS);
    let future = connectivity.has_pipe(Direction::ZPLUS);
    let valid_temporal = match signature.kind {
        BlockKind::Cube(_) | BlockKind::Walking(_) => return Ok(()),
        // Rotations use only temporal pipes; realignment owns any Hadamard flip.
        BlockKind::PatchRotation(_) => true,
        // Leaves inherit the patch orientation from their single temporal neighbour.
        BlockKind::Y | BlockKind::Port | BlockKind::Selective(_) => past ^ future,
        // Measurements consume a patch arriving from the past.
        BlockKind::Measurement(_) => past && !future,
        // T cultivation prepares a patch that escapes into the future.
        BlockKind::T => future && !past,
    };
    if !valid_temporal || connectivity.spatial_pipes().next().is_some() {
        return Err(CompileError::InvalidConnectivity {
            kind: signature.kind,
            connectivity,
        });
    }
    if missing_basis && !matches!(signature.kind, BlockKind::PatchRotation(_)) {
        return Err(CompileError::MissingBoundaryBasis {
            kind: signature.kind,
        });
    }
    Ok(())
}

pub(crate) fn compile_fixed_bulk(
    signature: BlockSignature,
    distance: u32,
) -> Result<CompiledTemplate, CompileError> {
    match signature.kind {
        BlockKind::Cube(kind) => compile_cube(
            kind,
            signature.connectivity,
            distance,
            signature
                .rounds
                .expect("cube signatures carry a resolved round count"),
            signature
                .layer_schedule
                .expect("cube signatures include a layer schedule"),
        ),
        BlockKind::Y => {
            let boundary_basis = signature
                .boundary_basis
                .expect("fixed-bulk validation requires a Y boundary basis");
            compile_y(boundary_basis, signature.connectivity, distance)
        }
        BlockKind::Measurement(basis) => {
            let boundary_basis = signature
                .boundary_basis
                .expect("fixed-bulk validation requires a measurement boundary basis");
            compile_measurement(basis, signature.connectivity, distance, boundary_basis)
        }
        BlockKind::Walking(kind) => compile_walking(kind, signature.connectivity, distance),
        BlockKind::PatchRotation(kind) => {
            compile_patch_rotation(kind, signature.connectivity, distance)
        }
        BlockKind::Port => {
            let boundary_basis = signature
                .boundary_basis
                .expect("fixed-bulk validation requires a port boundary basis");
            compile_port(boundary_basis, signature.connectivity, distance)
        }
        // `validate_fixed_bulk` admits `Selective`, but it is compiled to its two
        // per-basis arms by `compile_selective` (dispatched in `compile_signature`),
        // never through this single-template method.
        BlockKind::Selective(_) => unreachable!(
            "selective blocks are compiled via compile_selective, not compile_fixed_bulk"
        ),
        // `validate_fixed_bulk` admits `T`, but it is compiled to its cultivation +
        // escape template pair in `compile_signature` (the Selective
        // dispatch pattern), never through this single-template method.
        BlockKind::T => unreachable!(
            "T blocks are compiled via compile_signature's T arm, not compile_fixed_bulk"
        ),
    }
}

#[cfg(test)]
mod tests {
    use bloq_graph::Basis;

    use super::*;
    use crate::signature::Connectivity;

    fn t_signature(connectivity: Connectivity, boundary_basis: Option<Basis>) -> BlockSignature {
        BlockSignature {
            kind: BlockKind::T,
            rounds: Some(3),
            connectivity,
            boundary_basis,
            layer_schedule: None,
            surgery_side: Some(Direction::YMINUS),
        }
    }

    #[test]
    fn t_signature_requires_single_future_pipe_and_boundary_basis() {
        let zplus = Connectivity::ISOLATED.with_pipe(Direction::ZPLUS);
        validate_fixed_bulk(t_signature(zplus, Some(Basis::X))).unwrap();

        // A past pipe, an extra spatial pipe, or a missing basis are each rejected.
        let zminus = Connectivity::ISOLATED.with_pipe(Direction::ZMINUS);
        assert!(matches!(
            validate_fixed_bulk(t_signature(zminus, Some(Basis::X))),
            Err(CompileError::InvalidConnectivity {
                kind: BlockKind::T,
                connectivity,
            }) if connectivity == zminus
        ));
        assert!(matches!(
            validate_fixed_bulk(t_signature(
                zplus.with_pipe(Direction::XPLUS),
                Some(Basis::X),
            )),
            Err(CompileError::InvalidConnectivity {
                kind: BlockKind::T,
                ..
            })
        ));
        assert!(matches!(
            validate_fixed_bulk(t_signature(zplus, None)),
            Err(CompileError::MissingBoundaryBasis { kind: BlockKind::T })
        ));
    }

    #[test]
    fn fixed_measurement_requires_one_past_pipe_and_compiles_as_fixed() {
        let signature = |connectivity, boundary_basis| BlockSignature {
            kind: BlockKind::Measurement(Basis::X),
            rounds: None,
            connectivity,
            boundary_basis,
            layer_schedule: None,
            surgery_side: None,
        };
        let zminus = Connectivity::ISOLATED.with_pipe(Direction::ZMINUS);
        let valid = signature(zminus, Some(Basis::Z));

        validate_fixed_bulk(valid).expect("past-facing fixed measurement validates");
        compile_fixed_bulk(valid, 3).expect("fixed measurement compiles through the ordinary path");

        let zplus = Connectivity::ISOLATED.with_pipe(Direction::ZPLUS);
        assert!(matches!(
            validate_fixed_bulk(signature(zplus, Some(Basis::Z))),
            Err(CompileError::InvalidConnectivity { .. })
        ));
        assert!(matches!(
            validate_fixed_bulk(signature(zminus, None)),
            Err(CompileError::MissingBoundaryBasis {
                kind: BlockKind::Measurement(Basis::X),
            })
        ));
    }
}
