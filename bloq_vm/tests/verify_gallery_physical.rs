#![cfg(test)]

//! Physical checks for native non-Clifford gallery programs at distance 3.
//! The shared corpus and gallery categories select each program once, except
//! the ten-bit scaling case. The three-bit adder covers physical arithmetic;
//! the ten-bit adder retains compilation checks. Stim
//! covers Clifford transformations and distance behavior; focused tests below
//! cover execution hooks, prepared resources, and logical Choi states.
//! Analysis-only entries must fail compilation with their documented C0 limitation.
//!
//! T content and sampled CCZ resource positions must grow rank past 1.
//! Multi-resource programs use one magic input group per shot; complete
//! open-map Choi checks cover their joint input semantics.
//! Every shot must survive noiseless postselection with zero, constant detectors.
//! Branch-local detectors may be absent in untaken arms; at least one detector
//! must be evaluated in every shot to rule out vacuous success.
//!
use bloq_compile::{CompileConfig, CompileContext, CompileError};
use bloq_graph::GalleryCategory;
use bloq_test::{TestFixture, compile_fixtures, select_test_cases_for_fixture};
use bloq_vm::verify::VerifyReport;
use bloq_vm::{ExecError, PreparationContext, ShotContext, Simulator, run_bloq, run_bloq_with_io};

mod common;

use common::choi::exact::ExactChoi;
use common::choi::stabilizer::assert_stabilizer_state;

/// Shots per case. Noiseless post-selection never discards (see module docs),
/// so the factories need no larger sample than any other fixture.
const SHOTS: usize = 4;
const SEED: u64 = 0xC0FFEE;

fn input_seed(ctx: &PreparationContext<'_>, port: glam::IVec3) -> usize {
    ctx.inputs
        .iter()
        .find(|input| input.port == port)
        .unwrap_or_else(|| panic!("missing input seed {port:?}; have {:?}", ctx.inputs))
        .qubit
}

fn prepare_ccz(
    sim: &mut Simulator,
    ctx: &PreparationContext<'_>,
    ports: [glam::IVec3; 3],
) -> Result<(), ExecError> {
    let rank = sim.rank();
    ctx.prepare_ccz(ports)?;
    assert_eq!(
        sim.rank(),
        rank,
        "CCZ preparation stays deferred at the hook"
    );
    Ok(())
}

fn assert_basis_state(sim: &Simulator, ctx: &ShotContext<'_>, expected: &[(glam::IVec3, bool)]) {
    assert_eq!(ctx.outputs.len(), expected.len());
    let outputs = expected
        .iter()
        .map(|(port, _)| {
            let output = ctx
                .outputs
                .iter()
                .find(|output| output.port == *port)
                .unwrap();
            let frame = ctx.frames.iter().find(|frame| frame.port == *port).unwrap();
            (output, (frame.x.unwrap(), frame.z.unwrap()))
        })
        .collect::<Vec<_>>();
    let generators = expected
        .iter()
        .enumerate()
        .map(|(index, (_, value))| {
            let mut generator =
                bloq_vm::EnginePauliString::single(expected.len(), index, bloq_vm::EnginePauli::Z);
            generator.set_phase(2 * i32::from(*value));
            generator
        })
        .collect::<Vec<_>>();
    assert_stabilizer_state(sim, &outputs, &generators);
}

fn verify_fixture(fixture: TestFixture) -> Result<(), String> {
    let case = select_test_cases_for_fixture(fixture)
        .map_err(|error| error.to_string())?
        .into_iter()
        .find(|case| {
            case.metadata.origins.iter().any(|origin| {
                origin.fixture == fixture
                    && origin.fill_variant.is_none()
                    && !origin.flip_xz_basis
                    && origin.rotation.is_none()
            })
        })
        .ok_or_else(|| "no native case in the shared corpus".to_string())?;
    let program = case.build();
    let graph = program.flatten().map_err(|error| error.to_string())?;
    let offset = glam::IVec3::new(0, 0, -graph.spans().map_or(0, |(_, _, z)| *z.start()));
    let resources = program
        .interface
        .quantum_ports
        .iter()
        .filter(|port| {
            port.resource_type == "ccz" && port.direction == bloq_graph::PortDirection::Input
        })
        .map(|port| port.position + offset)
        .collect::<Vec<_>>();
    let (resources, remainder) = resources.as_chunks::<3>();
    assert!(
        remainder.is_empty(),
        "gallery CCZ inputs are complete triples"
    );
    let ctx = CompileContext::new(CompileConfig::new(3).with_prepare_t_with_mpps(true));
    let compiled = ctx.compile(&program);
    if fixture.in_category(GalleryCategory::AnalysisOnly) {
        return match compiled {
            Err(CompileError::BlockGraph(bloq_graph::BlockGraphError::Stabilizer(
                bloq_graph::StabilizerError::UnavailableControlParity { .. },
            ))) => Ok(()),
            Err(error) => Err(format!("wrong compilation limitation: {error}")),
            Ok(_) => Err("analysis-only gallery entry unexpectedly compiled".into()),
        };
    }
    let bloq = compiled.map_err(|e| format!("compile: {e}"))?.bloq;
    if matches!(
        fixture,
        TestFixture::CCZInjectedAnd | TestFixture::CCZInjectedMaj
    ) {
        bloq.validate()
            .map_err(|error| format!("validate: {error}"))?;
    }

    let report = if resources.is_empty() {
        run_bloq(&bloq, SHOTS, SEED)
    } else {
        run_bloq_with_io(
            &bloq,
            SHOTS,
            SEED,
            |sim, ctx| {
                // Joint resource semantics are checked by verify_choi's sweep.
                let group = ctx.shot * (resources.len() - 1) / (SHOTS - 1);
                prepare_ccz(sim, ctx, resources[group])
            },
            |_, _| Ok(()),
        )
    }
    .map_err(|error| format!("run_bloq: {error}"))?;

    assert_physical_invariants(&report)
}

/// The invariants that must hold for any noiseless physical run: zero discards,
/// a non-vacuous detector check, and the expected stabilizer-rank profile.
/// Sampled resource positions receive an ideal CCZ state; no case
/// receives a detector or rank exemption.
fn assert_physical_invariants(report: &VerifyReport) -> Result<(), String> {
    if report.shots != SHOTS {
        return Err(format!("ran {} shots, expected {SHOTS}", report.shots));
    }

    if report.discarded != 0 {
        return Err(format!(
            "{} shot(s) discarded; a noiseless run must never discard",
            report.discarded
        ));
    }
    if !report.all_detectors_constant() {
        let varied: Vec<_> = report
            .detectors
            .iter()
            .filter(|d| !d.constant)
            .map(|d| &d.per_shot)
            .collect();
        return Err(format!("a detector varied across shots: {varied:?}"));
    }
    if let Some((index, detector)) = report
        .detectors
        .iter()
        .enumerate()
        .find(|(_, detector)| detector.value == Some(true))
    {
        return Err(format!(
            "detector {index} has noiseless value 1 after sign normalization: {:?}",
            detector.per_shot
        ));
    }

    // Non-vacuous coverage: constancy must not pass because nothing was
    // evaluated. At least one detector must be evaluated in every shot — the
    // weak guard, since a conditionally registered detector is legitimately
    // evaluated only in the shots that select its component.
    if report.detectors.is_empty() {
        return Err("no detectors — vacuous constancy".to_string());
    }
    if !report
        .detectors
        .iter()
        .any(|d| d.per_shot.len() == report.shots)
    {
        return Err("no detector evaluated in every shot — vacuous constancy".to_string());
    }

    if report.max_rank <= 1 {
        return Err(format!(
            "non-Clifford fixture stayed at rank {} (expected > 1)",
            report.max_rank
        ));
    }
    Ok(())
}

#[test]
fn non_clifford_gallery_d3() {
    let mut count = 0;
    for fixture in compile_fixtures().iter().copied().filter(|fixture| {
        fixture.in_category(GalleryCategory::NonClifford) && *fixture != TestFixture::TenBitAdder
    }) {
        count += 1;
        verify_fixture(fixture).unwrap_or_else(|error| panic!("{}: {error}", fixture.id()));
    }
    assert!(count > 0, "non-Clifford fixture selection was empty");
}

fn assert_identity_choi(source: &str, input_port: glam::IVec3, output_port: glam::IVec3) {
    let graph = bloq_graph::BlockGraph::from_blog_text(source)
        .expect("identity graph parses")
        .fix_shadowed_faces();
    let bloq = CompileContext::new(CompileConfig::new(3).with_prepare_t_with_mpps(true))
        .compile(&graph)
        .expect("identity graph compiles")
        .bloq;
    ExactChoi::new(&[input_port], &[output_port], |_| Ok(())).run(&bloq, 2, SEED, |_, _| Ok(()));
}

#[test]
fn execution_hooks_handle_temporal_and_spatial_ports() {
    assert_identity_choi(
        "BLOG 1.0\n\n\
         \x20 0: Port [0,0,0]\n\
         \x20 1: ZXZ [0,0,1]\n\
         \x20 2: Port [0,0,2]\n\
         \x20 [0,0,0] -> +Z\n\
         \x20 [0,0,1] -> +Z\n",
        glam::ivec3(0, 0, 0),
        glam::ivec3(0, 0, 2),
    );
    assert_identity_choi(
        "BLOG 1.0\n\n\
         \x20 0: Port [-1,0,0] role=input\n\
         \x20 1: ZXZ [0,0,0]\n\
         \x20 2: Port [0,0,1]\n\
         \x20 [-1,0,0] -> +X\n\
         \x20 [0,0,0] -> +Z\n",
        glam::ivec3(-1, 0, 0),
        glam::ivec3(0, 0, 1),
    );
    assert_identity_choi(
        "BLOG 1.0\n\n\
         \x20 0: Port [0,0,0]\n\
         \x20 1: ZXZ [0,0,1]\n\
         \x20 2: Port [1,0,1] role=output\n\
         \x20 [0,0,0] -> +Z\n\
         \x20 [0,0,1] -> +X\n",
        glam::ivec3(0, 0, 0),
        glam::ivec3(1, 0, 1),
    );
}

#[test]
fn ccz_resource_hook_drives_gate_teleportation() {
    const RESOURCE: [glam::IVec3; 3] = [
        glam::IVec3::new(0, 2, 0),
        glam::IVec3::new(1, 2, 0),
        glam::IVec3::new(2, 2, 0),
    ];
    const OUTPUT: [glam::IVec3; 3] = [
        glam::IVec3::new(0, 0, 1),
        glam::IVec3::new(1, 0, 1),
        glam::IVec3::new(2, 0, 1),
    ];
    let graph = bloq_graph::GalleryItem::CCZGateTeleport.build();
    let bloq = CompileContext::new(CompileConfig::new(3).with_prepare_t_with_mpps(true))
        .compile(&graph)
        .expect("CCZ teleport compiles")
        .bloq;
    let oracle = ExactChoi::new(&OUTPUT, &OUTPUT, |sim| {
        sim.ccz(0, 1, 2)?;
        Ok(())
    });
    let report = oracle.run_prepared(
        &bloq,
        SHOTS,
        SEED,
        &RESOURCE,
        |sim, ctx| prepare_ccz(sim, ctx, RESOURCE),
        |_, _| Ok(()),
    );
    assert_physical_invariants(&report).expect("physical invariants");
}

#[test]
fn and_4t_choi_map_d3() {
    const X: glam::IVec3 = glam::IVec3::new(3, 0, 1);
    const Y: glam::IVec3 = glam::IVec3::new(3, 2, 1);
    const XY: glam::IVec3 = glam::IVec3::new(3, 0, 3);

    let program = bloq_graph::GalleryItem::And4T.build();
    let bloq = CompileContext::new(CompileConfig::new(3).with_prepare_t_with_mpps(true))
        .compile(&program)
        .expect("four-T AND module compiles")
        .bloq;
    assert_eq!(bloq.logical_outputs().len(), 3);
    let oracle = ExactChoi::new(&[X, Y], &[X, Y, XY], |sim| {
        sim.h(2);
        sim.ccz(0, 1, 2)?;
        sim.h(2);
        Ok(())
    });
    let report = oracle.run(&bloq, 32, SEED, |_, _| Ok(()));
    assert_eq!(report.shots, 32);
    assert_eq!(report.discarded, 0);
    assert!(report.max_rank > 1);
}

#[test]
fn ccz_injected_and_choi_map_d3() {
    const AND_SHOTS: usize = 32;
    const Q: glam::IVec3 = glam::IVec3::new(5, 2, 0);
    const I: glam::IVec3 = glam::IVec3::new(5, 1, 0);
    const CCZ: [glam::IVec3; 3] = [
        glam::IVec3::new(-1, 0, 0),
        glam::IVec3::new(-1, 1, 0),
        glam::IVec3::new(-1, 2, 0),
    ];
    const OUT: glam::IVec3 = glam::IVec3::new(0, 1, 2);

    let program = bloq_graph::GalleryItem::CCZInjectedAnd.build();
    let bloq = CompileContext::new(CompileConfig::new(3).with_prepare_t_with_mpps(true))
        .compile(&program)
        .expect("CCZ-injected AND module compiles")
        .bloq;
    let oracle = ExactChoi::new(&[Q, I], &[Q, I, OUT], |sim| {
        sim.h(2);
        sim.ccz(0, 1, 2)?;
        sim.h(2);
        Ok(())
    });
    let report = oracle.run_prepared(
        &bloq,
        AND_SHOTS,
        SEED,
        &CCZ,
        |sim, ctx| prepare_ccz(sim, ctx, CCZ),
        |_, _| Ok(()),
    );
    assert_eq!(report.discarded, 0);
    assert!(report.max_rank > 1);
}

#[test]
fn ccz_injected_maj_truth_table_d3() {
    const MAJ_SHOTS: usize = 8;
    const C: glam::IVec3 = glam::IVec3::new(2, -1, 1);
    const I: glam::IVec3 = glam::IVec3::new(0, 1, 0);
    const T: glam::IVec3 = glam::IVec3::new(5, 1, 4);
    const CCZ: [glam::IVec3; 3] = [
        glam::IVec3::new(-1, 0, 3),
        glam::IVec3::new(-1, 1, 3),
        glam::IVec3::new(-1, 2, 3),
    ];
    const OUTPUTS: [glam::IVec3; 5] = [
        glam::IVec3::new(2, 3, 1),
        glam::IVec3::new(3, 0, 6),
        glam::IVec3::new(3, 2, 6),
        glam::IVec3::new(4, 0, 6),
        glam::IVec3::new(4, 2, 6),
    ];

    let program = bloq_graph::GalleryItem::CCZInjectedMaj.build();
    let mut child = program.clone();
    child.name = "Maj".into();
    let mut body = bloq_graph::BlockGraph::new();
    let quantum_connections = child
        .interface
        .quantum_ports
        .iter()
        .map(|port| {
            let role = match port.direction {
                bloq_graph::PortDirection::Input => bloq_graph::PortRole::Input,
                bloq_graph::PortDirection::Output => bloq_graph::PortRole::Output,
            };
            body.try_add_block(
                bloq_graph::Block::new(port.position, bloq_graph::BlockKind::Port)
                    .with_port_role(role)
                    .unwrap(),
            )
            .unwrap();
            let endpoint = bloq_graph::InstancePort {
                instance: "maj".into(),
                port: port.name.clone(),
            };
            match port.direction {
                bloq_graph::PortDirection::Input => bloq_graph::QuantumConnection::Input {
                    block: port.position,
                    input: endpoint,
                    hadamard: false,
                },
                bloq_graph::PortDirection::Output => bloq_graph::QuantumConnection::Output {
                    output: endpoint,
                    block: port.position,
                    hadamard: false,
                },
            }
        })
        .collect();
    let root = bloq_graph::BlockGraph::definition(
        "main",
        body,
        child.interface.clone(),
        vec![bloq_graph::ModuleInstance {
            name: "maj".into(),
            definition: child.name.clone(),
            rotation: Default::default(),
            translation: glam::IVec3::ZERO,
        }],
        quantum_connections,
        Vec::new(),
    );
    let program = bloq_graph::BlockGraph::from_definitions(vec![child, root]).unwrap();
    let bloq = CompileContext::new(CompileConfig::new(3).with_prepare_t_with_mpps(true))
        .compile(&program)
        .expect("CCZ-injected MAJ module compiles")
        .bloq;
    let report = run_bloq_with_io(
        &bloq,
        MAJ_SHOTS,
        SEED,
        |sim, ctx| {
            for (bit, port) in [C, I, T].into_iter().enumerate() {
                let qubit = input_seed(ctx, port);
                sim.h(qubit);
                if ctx.shot & (1 << bit) != 0 {
                    sim.x(qubit);
                }
            }
            prepare_ccz(sim, ctx, CCZ)
        },
        |sim, ctx| {
            let c = ctx.shot & 1 != 0;
            let i = ctx.shot & 2 != 0;
            let t = ctx.shot & 4 != 0;
            let carry = (c && i) ^ (c && t) ^ (i && t);
            let expected = [carry, c, carry, c ^ i, c ^ t];
            assert_basis_state(
                sim,
                ctx,
                &OUTPUTS.into_iter().zip(expected).collect::<Vec<_>>(),
            );
            Ok(())
        },
    )
    .expect("CCZ-injected MAJ executes");
    assert_eq!(report.discarded, 0);
    assert!(report.max_rank > 1);
}

#[test]
fn uma_truth_table_d3() {
    const C: glam::IVec3 = glam::IVec3::new(0, 0, 0);
    const CARRY: glam::IVec3 = glam::IVec3::new(0, 2, 0);
    const C_XOR_I: glam::IVec3 = glam::IVec3::new(1, 0, 0);
    const C_XOR_T: glam::IVec3 = glam::IVec3::new(1, 2, 0);
    const OUT: glam::IVec3 = glam::IVec3::new(2, 0, 2);

    let program = bloq_graph::GalleryItem::UMA.build();
    let bloq = CompileContext::new(CompileConfig::new(3).with_prepare_t_with_mpps(true))
        .compile(&program)
        .expect("UMA module compiles")
        .bloq;
    bloq.validate().expect("UMA module validates");
    let report = run_bloq_with_io(
        &bloq,
        8,
        SEED,
        |sim, ctx| {
            let c = ctx.shot & 1 != 0;
            let i = ctx.shot & 2 != 0;
            let t = ctx.shot & 4 != 0;
            let carry = (c && i) ^ (c && t) ^ (i && t);
            for (port, value) in
                [C, CARRY, C_XOR_I, C_XOR_T]
                    .into_iter()
                    .zip([c, carry, c ^ i, c ^ t])
            {
                let qubit = input_seed(ctx, port);
                sim.h(qubit);
                if value {
                    sim.x(qubit);
                }
            }
            Ok(())
        },
        |sim, ctx| {
            let expected = (ctx.shot & 1 != 0) ^ (ctx.shot & 2 != 0) ^ (ctx.shot & 4 != 0);
            assert_basis_state(sim, ctx, &[(OUT, expected)]);
            Ok(())
        },
    )
    .expect("UMA executes");
    assert_eq!(report.discarded, 0);
    assert!(report.all_detectors_constant());
    assert_eq!(report.max_rank, 1);
}
