//! Seven in-place T blocks, four Z checks and four selective caps give CCZ|+++>.

use bloq_compile::{CompileConfig, CompileContext};
use bloq_graph::verify::{BoundaryOrder, LogicalVerifier, QuizxGraph};
use bloq_graph::{
    Action, BinaryOp, Block, BlockGraph, BlockKind, CubeKind, Direction, Expr, MeasureTarget, Pipe,
    PortRole, SelectiveKind,
};
use glam::IVec3;
use quizx::circuit::Circuit;
use quizx::graph::{BasisElem, GraphLike};

fn xor(vars: &[String]) -> Expr {
    let mut iter = vars.iter().map(|v| Expr::Var(v.clone()));
    let first = iter.next().expect("nonempty");
    iter.fold(first, |a, b| {
        Expr::Binary(BinaryOp::Xor, Box::new(a), Box::new(b))
    })
}

fn main() {
    // Parity label -> (x, y).
    let sites: [(usize, i32, i32); 7] = [
        (1, 0, 0),
        (2, 0, 2),
        (4, 1, 0),
        (7, 1, 2),
        (3, 2, 0),
        (5, 2, 2),
        (6, 3, 0),
    ];
    let pos = |s: usize| {
        sites
            .iter()
            .find(|t| t.0 == s)
            .map(|t| (t.1, t.2))
            .expect("hand-built CCZ layout is valid")
    };
    // Check, z layer.
    let checks: [(&str, &[usize], i32); 4] = [
        ("m1247", &[1, 2, 4, 7], 1),
        ("m356", &[3, 5, 6], 1),
        ("m347", &[3, 4, 7], 2),
        ("m145", &[1, 4, 5], 3),
    ];
    let top = 3;
    let mut g = BlockGraph::new();
    let mut actions = Vec::new();
    for &(_, x, y) in &sites {
        g.try_add_block(Block::new(IVec3::new(x, y, 0), BlockKind::T))
            .expect("hand-built CCZ layout is valid");
        for z in 1..=top {
            g.try_add_block(Block::new(
                IVec3::new(x, y, z),
                BlockKind::Cube(CubeKind::ZXX),
            ))
            .expect("hand-built CCZ layout is valid");
        }
        for z in 0..top {
            g.try_add_pipe(Pipe::new(IVec3::new(x, y, z), Direction::ZPLUS))
                .expect("hand-built CCZ layout is valid");
        }
    }
    for &(name, members, z) in &checks {
        let xs: Vec<i32> = members.iter().map(|&s| pos(s).0).collect();
        let (lo, hi) = (
            *xs.iter().min().expect("hand-built CCZ layout is valid"),
            *xs.iter().max().expect("hand-built CCZ layout is valid"),
        );
        for x in lo..=hi {
            g.try_add_block(Block::new(
                IVec3::new(x, 1, z),
                BlockKind::Cube(CubeKind::ZZX),
            ))
            .expect("hand-built CCZ layout is valid");
        }
        for x in lo..hi {
            g.try_add_pipe(Pipe::new(IVec3::new(x, 1, z), Direction::XPLUS))
                .expect("hand-built CCZ layout is valid");
        }
        for &s in members {
            let (x, y) = pos(s);
            let dir = if y == 0 {
                Direction::YMINUS
            } else {
                Direction::YPLUS
            };
            g.try_add_pipe(Pipe::new(IVec3::new(x, 1, z), dir))
                .expect("hand-built CCZ layout is valid");
        }
        let (x, y) = pos(members[0]);
        let dir = if y == 0 {
            Direction::YMINUS
        } else {
            Direction::YPLUS
        };
        actions.push(Action::Measure {
            target: MeasureTarget::Edge {
                src: IVec3::new(x, 1, z),
                dir,
            },
            name: name.into(),
        });
    }
    // Pivot selectors from c = H_P^{-1} s.
    let selectors: [(usize, &[&str]); 4] = [
        (5, &["m145"]),
        (7, &["m1247"]),
        (3, &["m347", "m1247"]),
        (6, &["m356", "m347", "m1247", "m145"]),
    ];
    for &(site, vars) in &selectors {
        let (x, y) = pos(site);
        let cap = IVec3::new(x, y, top + 1);
        g.try_add_block(Block::new(cap, BlockKind::Selective(SelectiveKind::XY)))
            .expect("hand-built CCZ layout is valid");
        g.try_add_pipe(Pipe::new(cap - IVec3::Z, Direction::ZPLUS))
            .expect("hand-built CCZ layout is valid");
        let c = xor(&vars.iter().map(|v| (*v).to_string()).collect::<Vec<_>>());
        // T: X when c = 0. T^dagger: X when c = 1.
        let dagger = (site as u32).count_ones().is_multiple_of(2);
        let condition = if dagger { c } else { Expr::Not(Box::new(c)) };
        actions.push(Action::Resolve {
            target: cap,
            condition,
        });
    }
    let mut outputs = Vec::new();
    for site in [1, 2, 4] {
        let (x, y) = pos(site);
        let port = IVec3::new(x, y, top + 1);
        g.try_add_block(
            Block::new(port, BlockKind::Port)
                .with_port_role(PortRole::Output)
                .expect("hand-built CCZ layout is valid"),
        )
        .expect("hand-built CCZ layout is valid");
        g.try_add_pipe(Pipe::new(port - IVec3::Z, Direction::ZPLUS))
            .expect("hand-built CCZ layout is valid");
        outputs.push(port);
    }
    g.set_actions_deferred(actions)
        .expect("hand-built CCZ layout is valid");
    g.validate_source().expect("source valid");
    println!("{}", g.to_blog_body_text());
    let verifier = LogicalVerifier::with_boundaries_and_internal_measurements(
        &g,
        BoundaryOrder::new(vec![], outputs),
    )
    .expect("verifier");
    let mut circuit = Circuit::new(3);
    circuit.add_gate("ccz", vec![0, 1, 2]);
    let mut expected: QuizxGraph = circuit.to_graph();
    expected.plug_inputs(&[BasisElem::X0; 3]);
    verifier
        .verify(&expected, 64, 0x5eed)
        .expect("7T CCZ verifies");
    println!("logical map verified");
    let compiled = CompileContext::new(CompileConfig::new(3).with_prepare_t_with_mpps(true))
        .compile_and_validate(&g)
        .expect("compiled");
    println!(
        "compiled, {} physical records",
        compiled.bloq.logical_outputs().len()
    );
}
