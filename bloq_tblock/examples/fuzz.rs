//! Check compiled circuits against their exact QuiZX maps.

use bloq_graph::verify::{BoundaryOrder, LogicalVerifier, QuizxGraph};
use bloq_interop::parse_qasm;
use bloq_tblock::{
    Deferral, MachineOptions, Options, Placement, Reduction, TMode, compile, metrics,
};

struct Rng(u64);
impl Rng {
    fn next(&mut self, n: u64) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0 % n
    }
}

fn random_qasm(k: u64, gates: u64, seed: u64) -> String {
    let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
    let mut s = format!("OPENQASM 2.0; include \"qelib1.inc\"; qreg q[{k}];\n");
    for _ in 0..gates {
        let a = rng.next(k);
        let mut b = rng.next(k);
        if b == a {
            b = (a + 1) % k;
        }
        let c = (b + 1 + rng.next(k - 1)) % k;
        if c != a && c != b && rng.next(12) == 0 {
            s.push_str(&format!("ccx q[{a}],q[{b}],q[{c}];\n"));
            continue;
        }
        s.push_str(&match rng.next(10) {
            0 | 1 => format!("h q[{a}];\n"),
            2 | 3 => format!("t q[{a}];\n"),
            4 => format!("tdg q[{a}];\n"),
            5 => format!("s q[{a}];\n"),
            6 | 7 => format!("cx q[{a}],q[{b}];\n"),
            8 => format!("cz q[{a}],q[{b}];\n"),
            _ => format!("x q[{a}];\n"),
        });
    }
    s
}

type Mode = (Deferral, Reduction, TMode, Placement);

static CHECKS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

fn check(source: &str, mode: Mode, seed: u64) -> Result<bloq_tblock::Metrics, String> {
    let circuit = parse_qasm(source)?;
    let options = Options {
        deferral: mode.0,
        reduction: mode.1,
        machine: MachineOptions {
            t_mode: mode.2,
            placement: mode.3,
            ..Default::default()
        },
    };
    let built = compile(&circuit, &options)?;
    if std::env::var_os("TRACE").is_some() {
        eprintln!("compiled {:?}", metrics(&built.graph));
    }
    built.graph.validate_source().map_err(|e| e.to_string())?;
    if std::env::var_os("TRACE").is_some() {
        eprintln!("validated");
    }
    if metrics(&built.graph).t_blocks > 12 {
        return Err("skip".into());
    }
    let verifier = LogicalVerifier::with_boundaries_and_internal_measurements(
        &built.graph,
        BoundaryOrder::new(built.inputs.clone(), built.outputs.clone()),
    )
    .map_err(|e| e.to_string())?;
    let expected: QuizxGraph = circuit.to_graph();
    verifier
        .verify(&expected, 24, seed)
        .map_err(|e| e.to_string())?;
    CHECKS.fetch_add(built.checks, std::sync::atomic::Ordering::Relaxed);
    Ok(metrics(&built.graph))
}

fn shrink(source: &str, deferral: Mode, seed: u64) -> String {
    let mut lines: Vec<String> = source.lines().map(str::to_owned).collect();
    let mut i = 1;
    while i < lines.len() {
        let mut candidate = lines.clone();
        candidate.remove(i);
        if check(&candidate.join("\n"), deferral, seed).is_err_and(|e| e != "skip") {
            lines = candidate;
        } else {
            i += 1;
        }
    }
    lines.join("\n")
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let count: u64 = args
        .first()
        .map_or(50, |s| s.parse().expect("numeric argument"));
    let only: Option<u64> = args.get(1).map(|s| s.parse().expect("numeric argument"));
    let mut failures = 0;
    let mut skipped = 0;
    let grid = Placement::Grid { columns: 0 };
    let modes = [
        (Deferral::Full, Reduction::Raw, TMode::Gadget, grid),
        (Deferral::None, Reduction::Raw, TMode::Gadget, grid),
        (Deferral::Full, Reduction::Zx, TMode::Gadget, grid),
        (
            Deferral::Full,
            Reduction::Raw,
            TMode::Gadget,
            Placement::Strip,
        ),
        (Deferral::Full, Reduction::Raw, TMode::Block, grid),
        (
            Deferral::Full,
            Reduction::Raw,
            TMode::Block,
            Placement::Strip,
        ),
    ];
    let mode_only: Option<usize> = std::env::var("MODE")
        .ok()
        .map(|m| m.parse().expect("numeric argument"));
    for (mode_index, deferral) in modes.into_iter().enumerate() {
        if mode_only.is_some_and(|m| m != mode_index) {
            continue;
        }
        for seed in 1..=count {
            let k = 2 + seed % 4;
            let source = random_qasm(k, 4 + seed % 20, seed);
            if only.is_some_and(|o| o != seed) {
                continue;
            }
            if only.is_some() {
                eprintln!("{source}");
            }
            if std::env::var_os("TRACE").is_some() {
                eprintln!("{deferral:?} {seed}");
            }
            let result = check(&source, deferral, seed);
            if result.as_ref().is_err_and(|e| e == "skip") {
                skipped += 1;
                continue;
            }
            if let Err(e) = result {
                failures += 1;
                if failures <= 4 {
                    println!(
                        "{deferral:?} seed {seed}: {e}\n{}",
                        shrink(&source, deferral, seed)
                    );
                }
            }
        }
    }
    println!(
        "failures {failures}, skipped {skipped} with more than 12 T, internal checks verified {}",
        CHECKS.load(std::sync::atomic::Ordering::Relaxed)
    );
}
