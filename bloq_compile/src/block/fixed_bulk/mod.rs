mod compiler;
mod grid;
pub(crate) mod observable;
pub(crate) mod port;
pub(super) mod realignment;
mod regular;
mod rotation;
mod selective;
mod spatial;
pub(super) mod spatial_hadamard;
pub(crate) mod t;
pub(super) mod utils;
mod ybasis;

use std::sync::Arc;

use bloq_circuit::{Chunk, ChunkOrLoop};
use bloq_graph::{Basis, CubeKind};
use bloq_ir::lowering::BloqTemplate;

use crate::CompileError;
use crate::block::fixed_bulk::observable::{
    INIT_CHUNK, build_gateway_from_patch, cube_operator_faces, resolve_gateway_measurements_at,
};
use crate::block::fixed_bulk::regular::build_regular_patch_with_reverse;
use crate::block::fixed_bulk::spatial::build_spatial_patch_with_reverse;
use crate::block::fixed_bulk::utils::{
    make_normal_surface_code_patch, make_surface_code_chunk, make_surface_code_cube_chunks,
};
use crate::block::measurements::stage_chunks;
use crate::block::patch::Patch;
use crate::block::{CompiledTemplate, LoweringTemplate};
use crate::signature::{Connectivity, LayerSchedule};

pub(crate) use compiler::{compile_fixed_bulk, validate_fixed_bulk};
pub(crate) use port::compile_multiplex_port;
pub(crate) use realignment::compile_realignment;
pub(crate) use selective::{SelectiveTemplates, compile_measurement, compile_selective};
pub(crate) use spatial_hadamard::{SpatialHadamardKey, WallSide, compile_spatial_hadamard};

/// Patch and collapsed-data maps for a cube-shaped block: spatial cubes get
/// the tqec spatial template, regular cubes the boundary-basis classifier.
/// `layer_schedule` is the CX-slot depth of the cube's z-layer spatial
/// component, which every member shares (see [`LayerSchedule`]).
fn build_cube_patch(
    kind: CubeKind,
    connectivity: Connectivity,
    distance: u32,
    layer_schedule: LayerSchedule,
) -> grid::PatchData {
    build_cube_patch_with_reverse(kind, connectivity, distance, layer_schedule, false)
}

fn build_cube_patch_with_reverse(
    kind: CubeKind,
    connectivity: Connectivity,
    distance: u32,
    layer_schedule: LayerSchedule,
    reverse_schedule: bool,
) -> grid::PatchData {
    if kind.is_spatial() {
        build_spatial_patch_with_reverse(
            kind,
            distance,
            connectivity,
            layer_schedule,
            reverse_schedule,
        )
    } else {
        build_regular_patch_with_reverse(
            kind,
            distance,
            connectivity,
            layer_schedule,
            reverse_schedule,
        )
    }
}

/// Compile a cube block into init + bulk loop + meas chunks.
fn compile_cube(
    kind: CubeKind,
    connectivity: Connectivity,
    distance: u32,
    rounds: u32,
    layer_schedule: LayerSchedule,
) -> Result<CompiledTemplate, CompileError> {
    let (patch, init_data, meas_data) =
        build_cube_patch(kind, connectivity, distance, layer_schedule);
    let cube = make_surface_code_cube_chunks(&patch, Some(&init_data), &meas_data)?;
    let (init, init_ids) = cube.init.expect("cube initialization was requested");
    let repetitions = cube_bulk_repetitions(rounds);

    let mut chunks = vec![ChunkOrLoop::Single(Box::new(init))];
    // The stage that ends up last is the one whose recorded index the gateway
    // needs; the extended schedule picks between the forward and reverse round.
    let meas_ids;
    if layer_schedule.is_extended() {
        let (reverse_patch, _, _) =
            build_cube_patch_with_reverse(kind, connectivity, distance, layer_schedule, true);
        let reverse = make_surface_code_cube_chunks(&reverse_patch, None, &meas_data)?;
        if repetitions >= 2 {
            chunks.push(ChunkOrLoop::Loop {
                body: vec![reverse.bulk.clone(), cube.bulk],
                repetitions: repetitions / 2,
            });
        }
        if repetitions % 2 == 1 {
            chunks.push(ChunkOrLoop::Single(Box::new(reverse.bulk)));
            chunks.push(ChunkOrLoop::Single(Box::new(cube.meas)));
            meas_ids = cube.meas_ids;
        } else {
            chunks.push(ChunkOrLoop::Single(Box::new(reverse.meas)));
            meas_ids = reverse.meas_ids;
        }
    } else {
        // A two-round cube — `height=d/2` at `d=3`, say — is all init and meas with
        // no bulk left over. Drop the loop rather than emit a zero-repetition
        // one, which downstream stage indexing treats as a real stage.
        if repetitions > 0 {
            chunks.push(ChunkOrLoop::Loop {
                body: vec![cube.bulk],
                repetitions,
            });
        }
        chunks.push(ChunkOrLoop::Single(Box::new(cube.meas)));
        meas_ids = cube.meas_ids;
    }

    let measurement_chunk = stage_chunks(&chunks).count() - 1;
    let observable_gateway = resolve_gateway_measurements_at(
        build_gateway_from_patch(kind, connectivity, distance, &patch),
        &chunks,
        measurement_chunk,
        [(INIT_CHUNK, init_ids), (measurement_chunk, meas_ids)]
            .into_iter()
            .collect(),
        |key| cube_operator_faces(kind, distance, key),
    );

    Ok(Arc::new(LoweringTemplate::from_chunks(
        chunks,
        observable_gateway,
    )?))
}

/// Compile a temporal seam's padding templates: `rounds` standard
/// syndrome-extraction rounds over a plain `d x d` surface code face named by
/// the basis on its `y`-normal edges, with no data init or readout.
///
/// The face is always on the compact four-slot schedule — idling a bare patch
/// has no merges and no walls to lengthen a round for. The first round stays
/// outside the loop so its seam-facing consumer flows survive into
/// `boundary_flows` (they close against the node below the seam); the loop's
/// top-open creators are the backend post-`REPEAT` lookback pattern (they close
/// against the node above). One template per entry of `rounds`, sharing the
/// patch's Theta(d^2) tile build across them.
pub(crate) fn compile_seam_padding_rounds<const N: usize>(
    y_face_basis: Basis,
    distance: u32,
    rounds: [u32; N],
) -> Result<[Arc<bloq_ir::lowering::BloqTemplate>; N], CompileError> {
    compile_patch_memory_padding_rounds(
        &make_normal_surface_code_patch(distance, y_face_basis),
        rounds,
    )
}

/// The cube-derived padding builder the seam builder replaced, kept only so
/// `seam_padding_matches_the_cube_derived_patch` can hold one to the other.
#[cfg(test)]
fn compile_memory_padding_rounds<const N: usize>(
    kind: CubeKind,
    connectivity: Connectivity,
    distance: u32,
    layer_schedule: LayerSchedule,
    rounds: [u32; N],
) -> Result<[Arc<bloq_ir::lowering::BloqTemplate>; N], CompileError> {
    let (patch, _init_data, _meas_data) =
        build_cube_patch(kind, connectivity, distance, layer_schedule);
    compile_patch_memory_padding_rounds(&patch, rounds)
}

/// Compile one padding template per entry of `rounds`. The patch's memory chunk
/// is identical across every round count — padding carries no init or
/// measurement data to vary it — so it is built once and cloned into each.
fn compile_patch_memory_padding_rounds<const N: usize>(
    patch: &Patch,
    rounds: [u32; N],
) -> Result<[Arc<bloq_ir::lowering::BloqTemplate>; N], CompileError> {
    if rounds.contains(&0) {
        return Err(CompileError::PaddingRoundsZero);
    }
    let round = make_surface_code_chunk(patch, None, None)?;
    let mut templates = Vec::with_capacity(N);
    for count in rounds {
        if count == 1 {
            templates.push(single_round_padding_template(&round));
            continue;
        }
        let chunks = vec![
            ChunkOrLoop::Single(Box::new(round.clone())),
            ChunkOrLoop::Loop {
                body: vec![round.clone()],
                repetitions: count - 1,
            },
        ];
        // A padding template routes no observables: the seam's logical operator
        // moves through it untouched, and observable evidence stays on the cubes.
        let template =
            LoweringTemplate::from_chunks(chunks, crate::block::gateway::ObservableGateway::new())?;
        templates.push(template.program_template);
    }
    let Ok(templates) = <[_; N]>::try_from(templates) else {
        unreachable!("one template pushed per requested round count")
    };
    Ok(templates)
}

fn single_round_padding_template(round: &Chunk) -> Arc<BloqTemplate> {
    assert!(
        round
            .flows
            .iter()
            .all(|flow| flow.start.is_empty() != flow.end.is_empty()),
        "a memory round has only boundary flows"
    );
    let boundary_flows = round.flows.clone();
    #[cfg(debug_assertions)]
    {
        let mut sorted = boundary_flows.clone();
        crate::block::compile::sort_boundary_flows(&mut sorted);
        debug_assert_eq!(boundary_flows, sorted);
    }
    Arc::new(BloqTemplate::with_parts(
        round.circuit.clone(),
        Vec::new(),
        Vec::new(),
        boundary_flows,
        Vec::new(),
    ))
}

/// Bulk-loop repetitions for a cube of `rounds` syndrome rounds: the rounds left
/// over once the initialization and measurement stages have taken one each.
///
/// Infallible because `signature::cube_rounds` has already rejected any height
/// that resolves below two rounds, and it is the only producer of the round
/// counts that reach here.
pub(super) fn cube_bulk_repetitions(rounds: u32) -> u32 {
    rounds
        .checked_sub(2)
        .expect("block signatures reject cubes with fewer than 2 rounds")
}

/// The repetition count of the template's bulk loop, or `None` when the
/// schedule has no loop (e.g. a flattened two-round cube).
///
/// Shared by the `fixed_bulk` and `ybasis` test modules.
#[cfg(test)]
fn loop_repetitions(template: &LoweringTemplate) -> Option<u32> {
    template
        .program_template
        .circuit
        .entry_top_level_repeats()
        .first()
        .copied()
}

#[cfg(test)]
fn split_ops_into_moments(ops: &[bloq_circuit::Op]) -> Vec<Vec<bloq_circuit::Op>> {
    let mut moments = vec![Vec::new()];
    for op in ops {
        if matches!(op, bloq_circuit::Op::Tick) {
            moments.push(Vec::new());
        } else {
            moments
                .last_mut()
                .expect("at least one moment")
                .push(op.clone());
        }
    }
    moments
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;
    use crate::signature::BlockSignature;

    /// The schedule an isolated cube's own layer demands: a spatial cube is its
    /// own component, so it forces the padded depth on itself.
    fn isolated_layer_schedule(kind: CubeKind) -> LayerSchedule {
        if kind.is_spatial() {
            LayerSchedule::Padded
        } else {
            LayerSchedule::Compact
        }
    }

    fn port_signature() -> BlockSignature {
        BlockSignature {
            kind: bloq_graph::BlockKind::Port,
            rounds: None,
            connectivity: Connectivity::ISOLATED,
            boundary_basis: None,
            layer_schedule: None,
            surgery_side: None,
        }
    }

    fn temporal_port_signature(
        pipe_dir: bloq_graph::Direction,
        boundary_basis: bloq_graph::Basis,
    ) -> BlockSignature {
        BlockSignature {
            kind: bloq_graph::BlockKind::Port,
            rounds: None,
            connectivity: Connectivity::ISOLATED.with_pipe(pipe_dir),
            boundary_basis: Some(boundary_basis),
            layer_schedule: None,
            surgery_side: None,
        }
    }

    fn compile_fixture(
        kind: CubeKind,
        connectivity: Connectivity,
        distance: u32,
    ) -> Arc<LoweringTemplate> {
        // An unmodified cube is `height=d`, i.e. `distance` rounds.
        compile_cube(
            kind,
            connectivity,
            distance,
            distance,
            isolated_layer_schedule(kind),
        )
        .unwrap()
    }

    #[test]
    fn cube_zxz_d3_circuit_snapshot() {
        // Pins the emitted syndrome-round circuits byte-for-byte (including the
        // hash-order data-collapse batches) so round-emitter refactors stay
        // behavior-identical.
        let template = compile_fixture(CubeKind::ZXZ, Connectivity::ISOLATED, 3);
        insta::assert_snapshot!(
            "cube_zxz_d3_circuit",
            template.program_template.circuit.to_string()
        );
    }

    #[test]
    fn extended_layer_cube_rounds_are_six_slots_deep() {
        // Reset + 6 CX slots + measure is the 8-moment round the wall pipe's
        // GHZ bracket needs; every cube in the layer must match it.
        for schedule in [LayerSchedule::Extended, LayerSchedule::ExtendedY] {
            let (patch, _, _) =
                build_cube_patch(CubeKind::XZX, Connectivity::ISOLATED, 3, schedule);
            assert!(
                patch
                    .tiles()
                    .iter()
                    .all(|tile| tile.data_slots().len() == 6)
            );
        }
    }

    #[rstest]
    #[case(3, 3, None)]
    #[case(5, 5, Some(3))]
    #[case(5, 10, Some(8))]
    #[case(5, 3, None)]
    fn test_compile_cube_bulk_loop_follows_the_resolved_round_count(
        #[case] distance: u32,
        #[case] rounds: u32,
        #[case] repetitions: Option<u32>,
    ) {
        let template = compile_cube(
            CubeKind::ZXZ,
            Connectivity::ISOLATED,
            distance,
            rounds,
            LayerSchedule::Compact,
        )
        .expect("cube compiles");
        assert_eq!(loop_repetitions(&template), repetitions);
        if repetitions.is_none() {
            assert!(template.program_template.repeat_states.is_empty());
        }
    }

    #[test]
    fn memory_padding_single_round_is_all_boundary() {
        // One bulk round closes nothing internally: every stabilizer leaves a
        // seam-facing consumer chain and a top-open creator chain.
        let [template] =
            compile_seam_padding_rounds(Basis::X, 3, [1]).expect("single-round padding compiles");
        // d3 regular patch has d^2 - 1 = 8 stabilizer tiles.
        assert_eq!(template.boundary_flows.len(), 16);
        assert!(template.detectors.is_empty());
        assert!(template.repeat_states.is_empty());
    }

    #[test]
    fn memory_padding_multi_round_loops_the_tail() {
        let [template] =
            compile_seam_padding_rounds(Basis::X, 3, [4]).expect("multi-round padding compiles");
        let circuit = &template.circuit;
        let repetitions = circuit
            .body(circuit.entry_body())
            .expect("entry body exists")
            .ops()
            .iter()
            .find_map(|op| match op {
                bloq_circuit::Op::Repeat { repetitions, .. } => Some(*repetitions),
                _ => None,
            });
        assert_eq!(repetitions, Some(3), "first round stays outside the loop");
        assert_eq!(template.repeat_states.len(), 8);
        // The seam interface is unchanged by the loop: 8 bottom consumers from
        // the first round, 8 top-open loop creators.
        assert_eq!(template.boundary_flows.len(), 16);
    }

    #[test]
    fn memory_padding_zero_rounds_is_rejected() {
        assert!(matches!(
            compile_seam_padding_rounds(Basis::X, 3, [0]),
            Err(CompileError::PaddingRoundsZero)
        ));
    }

    /// A seam pads the same whichever builder resolves it, so a non-cube
    /// endpoint can fall back on the plain face without inventing a patch.
    #[test]
    fn seam_padding_matches_the_cube_derived_patch() {
        for kind in [CubeKind::XZZ, CubeKind::ZXX, CubeKind::ZXZ, CubeKind::XZX] {
            assert_ne!(kind.x(), kind.y(), "{kind:?} admits a temporal pipe");
            for distance in [3, 5] {
                let [cube] = compile_memory_padding_rounds(
                    kind,
                    Connectivity::ISOLATED,
                    distance,
                    LayerSchedule::Compact,
                    [1],
                )
                .expect("cube padding compiles");
                let [seam] = compile_seam_padding_rounds(kind.y(), distance, [1])
                    .expect("seam padding compiles");
                assert_eq!(
                    format!("{cube:?}"),
                    format!("{seam:?}"),
                    "{kind:?} at d{distance}",
                );
            }
        }
    }

    #[test]
    fn test_fixed_bulk_compiler_isolated_port_is_rejected() {
        // A port with no temporal pipe is not a valid leaf boundary.
        let sig = port_signature();
        let error = validate_fixed_bulk(sig).expect_err("isolated port should be rejected");
        assert!(matches!(
            error,
            CompileError::InvalidConnectivity {
                kind: bloq_graph::BlockKind::Port,
                ..
            }
        ));
    }

    #[rstest]
    fn test_fixed_bulk_compiler_spatial_port_is_rejected(
        #[values(bloq_graph::Direction::XPLUS, bloq_graph::Direction::YMINUS)]
        spatial_dir: bloq_graph::Direction,
    ) {
        let sig = BlockSignature {
            kind: bloq_graph::BlockKind::Port,
            rounds: None,
            connectivity: Connectivity::ISOLATED
                .with_pipe(bloq_graph::Direction::ZPLUS)
                .with_pipe(spatial_dir),
            boundary_basis: Some(bloq_graph::Basis::Z),
            layer_schedule: None,
            surgery_side: None,
        };
        let error = validate_fixed_bulk(sig).expect_err("spatial port should be rejected");
        assert!(matches!(
            error,
            CompileError::InvalidConnectivity {
                kind: bloq_graph::BlockKind::Port,
                ..
            }
        ));
    }

    #[rstest]
    fn test_fixed_bulk_compiler_temporal_port_dispatch(
        #[values(bloq_graph::Direction::ZPLUS, bloq_graph::Direction::ZMINUS)]
        pipe_dir: bloq_graph::Direction,
        #[values(bloq_graph::Basis::X, bloq_graph::Basis::Z)] boundary_basis: bloq_graph::Basis,
    ) {
        let sig = temporal_port_signature(pipe_dir, boundary_basis);
        validate_fixed_bulk(sig).expect("temporal port validates");
        let template = compile_fixed_bulk(sig, 3).unwrap();
        assert!(template.program_template.circuit.num_measurements() > 0);
        assert_eq!(template.observable_gateway.len(), 2);
    }

    #[rstest]
    fn test_cube_kinds_compile(
        #[values(
            CubeKind::XXZ,
            CubeKind::XZX,
            CubeKind::XZZ,
            CubeKind::ZXX,
            CubeKind::ZXZ,
            CubeKind::ZZX
        )]
        kind: CubeKind,
        #[values(3, 5)] d: u32,
    ) {
        compile_fixture(kind, Connectivity::ISOLATED, d);
    }

    #[test]
    fn test_compile_cube_zxz_d3_has_template_detector_side_tables() {
        let template = compile_fixture(CubeKind::ZXZ, Connectivity::ISOLATED, 3);
        assert!(template.program_template.circuit.num_measurements() > 0);
        assert!(!template.program_template.detectors.is_empty());
    }
}
