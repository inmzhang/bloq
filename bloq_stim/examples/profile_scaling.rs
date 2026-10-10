//! Module pipeline scaling probe. Usage:
//! `profile_scaling FAMILY SIZE DISTANCE ITERATIONS [pipeline|summary|build|link]`
//! Families: wire (direct seams), measured (parent connectors), wide (parallel wires),
//! wire-tree and measured-tree (balanced hierarchies; SIZE must be a power of two ≥2).
//! wire-ordered uses zero-padded names to isolate frame equation ordering.
//! Pipeline prints one JSON record per iteration; profile modes isolate repeated work.
//! BLOQ_PROFILE_READY=1 pauses after profile-mode setup until stdin receives a line.

use std::fmt::Write;
use std::hint::black_box;
use std::time::Instant;

use bloq_compile::{CompileConfig, CompileContext};
use bloq_graph::{BlockGraph, ModuleCertificationLimits};

const WIRE: &str = "module Stage {\n  in q_in: data = 0\n  out q_out: data = 2\n  0: Port [0,0,0] role=input <q_in>\n  1: XZX [0,0,1]\n  2: Port [0,0,2] role=output <q_out>\n  0 -> +Z\n  1 -> +Z\n";

fn source(family: &str, size: usize) -> String {
    assert!(size > 0);
    assert!(
        matches!(
            family,
            "wire" | "measured" | "wide" | "wire-tree" | "measured-tree" | "wire-ordered"
        ),
        "unknown scaling family: {family}"
    );
    let tree = family.ends_with("-tree");
    let family = family.strip_suffix("-tree").unwrap_or(family);
    let ordered = family.ends_with("-ordered");
    let family = family.strip_suffix("-ordered").unwrap_or(family);
    let site = |index| {
        if ordered {
            format!("s{index:06}")
        } else {
            format!("s{index}")
        }
    };
    let mut text = format!("BLOG 1.0\n{WIRE}");
    if family == "measured" {
        text.push_str("  3: ZXZ [1,0,0]\n  4: Z [1,0,1]\n  3 -> +Z\n  m = measure 4\n");
    }
    text.push_str("}\n");
    if tree {
        assert!(size >= 2 && size.is_power_of_two());
        for level in 1..=size.ilog2() {
            let child = if level == 1 {
                "Stage".into()
            } else {
                format!("M{}", level - 1)
            };
            let name = if level == size.ilog2() {
                "main".into()
            } else {
                format!("M{level}")
            };
            let length = 1usize << level;
            let shift = if family == "measured" {
                length
            } else {
                length / 2
            };
            let end = if family == "measured" {
                2 * length
            } else {
                length + 1
            };
            writeln!(text, "module {name} {{\n  in q_in: data = 100\n  out q_out: data = 101\n  a: {child} @ [0,0,0]\n  b: {child} @ [0,0,{shift}]\n  100: Port [0,0,0] role=input <q_in>\n  101: Port [0,0,{end}] role=output <q_out>\n  100 -> a.q_in\n  b.q_out -> 101").expect("writing to a String is infallible");
            if family == "measured" {
                writeln!(
                    text,
                    "  200: XZX [0,0,{shift}]\n  a.q_out -> 200\n  200 -> b.q_in"
                )
                .expect("writing to a String is infallible");
            } else {
                text.push_str("  a.q_out -> b.q_in\n");
            }
            text.push_str("}\n");
        }
        return text;
    }
    text.push_str("module main {\n");
    if family == "wide" {
        for index in 0..size {
            let input = 2 * index;
            let output = input + 1;
            let x = 3 * index;
            writeln!(text, "  in i{index}: data = {input}\n  out o{index}: data = {output}\n  s{index}: Stage @ [{x},0,0]\n  {input}: Port [{x},0,0] role=input <i{index}>\n  {output}: Port [{x},0,2] role=output <o{index}>\n  {input} -> s{index}.q_in\n  s{index}.q_out -> {output}").expect("writing to a String is infallible");
        }
    } else {
        let stride = if family == "measured" { 2 } else { 1 };
        let end = (size - 1) * stride + 2;
        writeln!(text, "  in q_in: data = 100\n  out q_out: data = 101\n  100: Port [0,0,0] role=input <q_in>\n  101: Port [0,0,{end}] role=output <q_out>\n  100 -> {}.q_in\n  {}.q_out -> 101", site(0), site(size - 1)).expect("writing to a String is infallible");
        for index in 0..size {
            let current = site(index);
            writeln!(text, "  {current}: Stage @ [0,0,{}]", index * stride)
                .expect("writing to a String is infallible");
            if index + 1 < size {
                let next = index + 1;
                let next_site = site(next);
                if family == "measured" {
                    let connector = 200 + index;
                    writeln!(text, "  {connector}: XZX [0,0,{}]\n  {current}.q_out -> {connector}\n  {connector} -> {next_site}.q_in", next * stride).expect("writing to a String is infallible");
                } else {
                    writeln!(text, "  {current}.q_out -> {next_site}.q_in")
                        .expect("writing to a String is infallible");
                }
            }
        }
    }
    text.push_str("}\n");
    text
}

fn timed<T>(run: impl FnOnce() -> T) -> (T, f64) {
    let start = Instant::now();
    let result = black_box(run());
    (result, start.elapsed().as_secs_f64() * 1_000.0)
}

fn main() {
    let args = std::env::args().collect::<Vec<_>>();
    assert!(args.len() >= 5, "FAMILY SIZE DISTANCE ITERATIONS [MODE]");
    let family = args[1].as_str();
    let size = args[2].parse::<usize>().expect("positive size");
    let distance = args[3].parse::<u32>().expect("odd code distance");
    let iterations = args[4].parse::<usize>().expect("positive iteration count");
    assert!(iterations > 0);
    let mode = args.get(5).map_or("pipeline", String::as_str);
    let text = source(family, size);
    let config = CompileConfig::new(distance);
    let (program, parse_ms) = timed(|| BlockGraph::from_text(&text).expect("valid scaling source"));
    if mode != "pipeline" {
        let context = CompileContext::new(config);
        let object =
            (mode == "link").then(|| context.compile_object(&program).expect("build module"));
        if std::env::var_os("BLOQ_PROFILE_READY").is_some() {
            eprintln!("profile-ready");
            std::io::stdin()
                .read_line(&mut String::new())
                .expect("profile start signal");
        }
        let (_, elapsed_ms) = timed(|| {
            for _ in 0..iterations {
                match mode {
                    "summary" => {
                        black_box(
                            program
                                .summarize_root(ModuleCertificationLimits::DEFAULT)
                                .expect("summarize module"),
                        );
                    }
                    "build" => {
                        black_box(
                            CompileContext::new(config)
                                .compile_object(&program)
                                .expect("build module"),
                        );
                    }
                    "link" => {
                        black_box(
                            context
                                .link_object(
                                    object
                                        .as_ref()
                                        .expect("link mode builds a module object before timing"),
                                )
                                .expect("link module"),
                        );
                    }
                    _ => panic!("expected pipeline, summary, build or link"),
                }
            }
        });
        eprintln!(
            "{family} n={size} d={distance} {mode}: {:.3} ms/iteration",
            elapsed_ms / iterations as f64
        );
        return;
    }
    for iteration in 0..iterations {
        let (summary, summary_ms) = timed(|| {
            program
                .summarize_root(ModuleCertificationLimits::DEFAULT)
                .expect("summarize module")
        });
        let boundary_rows = summary.boundary_rows().len();
        drop(summary);
        let context = CompileContext::new(config);
        let (object, build_ms) = timed(|| context.compile_object(&program).expect("build module"));
        let (compiled, link_ms) = timed(|| context.link_object(&object).expect("link module"));
        let (_, warm_build_ms) = timed(|| context.compile_object(&program).expect("cached module"));
        let bloq = compiled.bloq;
        let (_, validate_ms) = timed(|| bloq.validate().expect("valid compiled IR"));
        let (binary, binary_ms) = timed(|| bloq.to_binary());
        let (stim, stim_ms) =
            timed(|| bloq_stim::emit_bloq_stim(&bloq).expect("Clifford scaling circuit"));
        println!(
            "{{\"family\":\"{family}\",\"size\":{size},\"distance\":{distance},\"iteration\":{iteration},\"source_bytes\":{},\"parse_ms\":{parse_ms},\"summary_ms\":{summary_ms},\"build_ms\":{build_ms},\"link_ms\":{link_ms},\"warm_build_ms\":{warm_build_ms},\"validate_ms\":{validate_ms},\"binary_ms\":{binary_ms},\"stim_ms\":{stim_ms},\"boundary_rows\":{boundary_rows},\"quantum_nodes\":{},\"templates\":{},\"measurements\":{},\"binary_bytes\":{},\"stim_bytes\":{}}}",
            text.len(),
            bloq.quantum_node_count(),
            bloq.templates().len(),
            bloq.measurement_count(),
            binary.len(),
            stim.len()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scaling_families_link_repeatably_to_valid_ir() {
        for family in [
            "wire",
            "measured",
            "wide",
            "wire-tree",
            "measured-tree",
            "wire-ordered",
        ] {
            let program = BlockGraph::from_text(&source(family, 4)).unwrap();
            let base = family.split('-').next().unwrap();
            let reference = BlockGraph::from_text(&source(base, 4))
                .unwrap()
                .flatten()
                .unwrap();
            let graph = program.flatten().unwrap();
            // Hierarchy and instance spelling change; the physical source stays fixed.
            assert_eq!(
                graph.blocks().collect::<std::collections::HashSet<_>>(),
                reference.blocks().collect::<std::collections::HashSet<_>>()
            );
            assert_eq!(
                graph.pipes().collect::<std::collections::HashSet<_>>(),
                reference.pipes().collect::<std::collections::HashSet<_>>()
            );
            assert_eq!(graph.actions().len(), reference.actions().len());
            let context = CompileContext::new(CompileConfig::default());
            let object = context.compile_object(&program).unwrap();
            let first = context.link_object(&object).unwrap().bloq;
            let second = context.link_object(&object).unwrap().bloq;
            first.validate().unwrap();
            assert_eq!(first.to_binary(), second.to_binary());
            assert_eq!(
                first.output_frames().len(),
                if family == "wide" { 4 } else { 1 }
            );
            assert!(!bloq_stim::emit_bloq_stim(&first).unwrap().is_empty());
        }
    }
}
