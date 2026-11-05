//! Print compile figures for QASM files.
use bloq_interop::parse_qasm;
use bloq_tblock::{Deferral, Options, Reduction, compile, lower, metrics, reduce};

fn main() {
    for path in std::env::args().skip(1) {
        let name = std::path::Path::new(&path)
            .file_stem()
            .expect("stats input is valid")
            .to_string_lossy()
            .to_string();
        let source = std::fs::read_to_string(&path).expect("stats input is valid");
        let Ok(circuit) = parse_qasm(&source) else {
            println!("{name}: parse failed");
            continue;
        };
        let reduction = match std::env::var("REDUCTION").as_deref() {
            Ok("zx") => Reduction::Zx,
            _ => Reduction::Raw,
        };
        let program =
            lower(&reduce(&circuit, reduction), Deferral::Full).expect("stats input is valid");
        let batches = program.batches();
        let sites: usize = batches.iter().map(|b| b.1).sum();
        let ranks: usize = batches.iter().map(|b| b.2).sum();
        let max = batches.iter().map(|b| b.1).max().unwrap_or(0);
        let start = std::time::Instant::now();
        let built = compile(
            &circuit,
            &Options {
                reduction,
                ..Default::default()
            },
        )
        .expect("stats input is valid");
        let m = metrics(&built.graph);
        println!(
            "{name:24} k={:3} T={sites:4} rank={ranks:4} batches={:3} maxbatch={max:3} ops={:4} {:?} V={:7} occ={:6} sel={:4} {:.3}s",
            program.cols,
            batches.len(),
            program.ops.len(),
            m.extents,
            m.volume,
            m.occupied,
            m.selective,
            start.elapsed().as_secs_f64(),
        );
    }
}
