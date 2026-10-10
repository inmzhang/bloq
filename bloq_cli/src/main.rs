//! The `bloq` command-line binary.
//!
//! A thin shim over [`bloq_cli::run`], which the Python wheel's console script
//! also calls, so both entry points share one argument parser and exit-status
//! contract.

fn main() -> std::process::ExitCode {
    bloq_cli::run(std::env::args_os()).into()
}
