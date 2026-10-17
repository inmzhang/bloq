//! CLI for inspecting test cases and benchmark workloads.

use std::env;

use color_eyre::eyre::{Result, bail};

use bloq_test::{
    benchmark::{DEFAULT_COMPILE_DISTANCES, benchmark_value, case_slug},
    compile_ready_test_cases, find_test_case,
};

fn main() -> Result<()> {
    color_eyre::install()?;
    run()
}

fn run() -> Result<()> {
    let mut args = env::args().skip(1);
    match args.next().as_deref() {
        Some("list-scenarios") => {
            let args = args.collect::<Vec<_>>();
            let filter = parse_optional_flag(&args, "--filter")?;
            let case_query = parse_optional_flag(&args, "--case-query")?;
            let filter = filter.map(str::trim).filter(|s| !s.is_empty());
            for prefix in ["compile/runtime", "backend/stim"] {
                for &distance in DEFAULT_COMPILE_DISTANCES {
                    let id = format!("{prefix}/{}", benchmark_value(case_query, distance));
                    let matches = filter.is_none_or(|filter| match filter.strip_prefix('^') {
                        Some(prefix) => id.starts_with(prefix),
                        None => id.contains(filter),
                    });
                    if matches {
                        println!("{id}");
                    }
                }
            }
            Ok(())
        }
        Some("list-cases") => {
            let args = args.collect::<Vec<_>>();
            let filter = parse_optional_flag(&args, "--filter")?;
            let compile_ready = args.iter().any(|arg| arg == "--compile-ready");
            let slugs = args.iter().any(|arg| arg == "--slugs");
            let cases = if compile_ready || filter.is_some() {
                compile_ready_test_cases(filter)?
            } else {
                bloq_test::all_test_cases()?
            };
            for case in cases {
                if slugs {
                    println!("{}", case_slug(case.id()));
                } else {
                    println!("{}", case.id());
                }
            }
            Ok(())
        }
        Some("show-case") => {
            let Some(name) = args.next() else {
                bail!("usage: cargo run -p xtask -- show-case <exact-case-name>");
            };
            let case = find_test_case(&name)?;
            println!("name: {}", case.id());
            println!("aliases: {}", case.aliases().join(", "));
            println!("fixtures: {}", case.metadata.fixture_ids().join(", "));
            Ok(())
        }
        _ => {
            bail!(
                "usage: cargo run -p xtask -- <list-scenarios|list-cases|show-case> [--filter <scenario-id-substring>] [--case-query <fuzzy-case-query>] [--compile-ready] [--slugs] [args]"
            )
        }
    }
}

fn parse_optional_flag<'a>(args: &'a [String], flag: &str) -> Result<Option<&'a str>> {
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        if arg == flag {
            let Some(value) = iter.next() else {
                bail!("missing value for {flag}");
            };
            return Ok(Some(value));
        }
    }
    Ok(None)
}
