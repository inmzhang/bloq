//! Print one compiled source as BLOG.
use bloq_interop::parse_qasm;
use bloq_tblock::{MachineOptions, Options, Placement, TMode, compile, metrics};

fn main() {
    let source = std::env::args()
        .nth(1)
        .expect("one source argument that compiles");
    let circuit = parse_qasm(&format!("OPENQASM 2.0; include \"qelib1.inc\"; {source}"))
        .expect("one source argument that compiles");
    let mut options = Options::default();
    if std::env::var_os("BLOCK").is_some() {
        options.machine.t_mode = TMode::Block;
    }
    if std::env::var_os("STRIP").is_some() {
        options.machine.placement = Placement::Strip;
    }
    let _ = MachineOptions::default();
    let built = compile(&circuit, &options).expect("one source argument that compiles");
    eprintln!("checks {}", built.checks);
    println!("{}", built.graph.to_blog_body_text());
    println!("{:?}", metrics(&built.graph));
    println!("{:?}", built.graph.validate_source());
}
