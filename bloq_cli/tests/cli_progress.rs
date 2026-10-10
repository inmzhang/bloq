use std::process::Command;

fn assert_finished_once(stderr: &str) {
    assert_eq!(stderr.matches("Finished").count(), 1, "{stderr}");
    let footer = stderr.lines().last().expect("final status line");
    let elapsed = footer
        .split_once("compilation in ")
        .expect("elapsed-time footer")
        .1
        .strip_suffix('s')
        .expect("seconds suffix")
        .parse::<f64>()
        .expect("numeric elapsed time");
    assert!(elapsed.is_finite() && elapsed >= 0.0, "{footer}");
}

#[test]
fn view_renders_saved_ir_codecs_with_classical_flag_and_protects_input() {
    let directory = tempfile::tempdir().unwrap();
    let program = bloq_compile::compile(&bloq_graph::GalleryItem::T.build(), 3).unwrap();
    for (name, bytes) in [
        ("program.bloqir", program.to_text().into_bytes()),
        ("program.bloq", program.to_binary()),
    ] {
        let input = directory.path().join(name);
        std::fs::write(&input, &bytes).unwrap();
        let output = directory.path().join("graph.svg");
        for include_classical in [false, true] {
            let mut command = Command::new(env!("CARGO_BIN_EXE_bloq"));
            command
                .arg("view")
                .arg(&input)
                .args(["--svg", "--quiet", "-o"])
                .arg(&output);
            if include_classical {
                command.arg("--include-classical");
            }
            let result = command.output().unwrap();
            assert!(
                result.status.success(),
                "{}",
                String::from_utf8_lossy(&result.stderr)
            );
            assert!(result.stderr.is_empty());
            assert_eq!(
                std::fs::read_to_string(&output).unwrap(),
                program.to_svg(include_classical)
            );
        }
        let result = Command::new(env!("CARGO_BIN_EXE_bloq"))
            .arg("view")
            .arg(&input)
            .arg("-o")
            .arg(&input)
            .output()
            .unwrap();
        assert!(!result.status.success());
        assert!(String::from_utf8_lossy(&result.stderr).contains("aliases the input file"));
        assert_eq!(std::fs::read(&input).unwrap(), bytes);
    }
}

#[test]
fn stats_prints_static_inventory_for_both_codecs_without_full_audit() {
    let directory = tempfile::tempdir().unwrap();
    let mut program = bloq_ir::Bloq::new();
    // Missing region predicate wiring is invalid, but a static read needs no
    // semantic audit and does not flatten this dynamic program.
    let mut body = bloq_ir::SubGraph::new();
    body.add_node(bloq_ir::BloqNode::from_members(vec![]));
    program.add_node(bloq_ir::BloqNode::region(
        bloq_ir::RegionNode::RepeatUntilSuccess {
            restart_condition: bloq_ir::ClassicalExpr::In(0),
            restart_source: None,
            body,
        },
    ));
    assert!(program.validate().is_err());
    for (name, bytes) in [
        ("program.bloqir", program.to_text().into_bytes()),
        ("program.bloq", program.to_binary()),
    ] {
        let input = directory.path().join(name);
        std::fs::write(&input, bytes).unwrap();
        let result = Command::new(env!("CARGO_BIN_EXE_bloq"))
            .arg("stats")
            .arg(input)
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        assert_eq!(
            String::from_utf8(result.stdout).unwrap(),
            format!("{}\n", program.stats().unwrap())
        );
        assert!(result.stderr.is_empty());
    }
}

#[test]
fn redirected_progress_stays_on_stderr_and_quiet_suppresses_it() {
    let run = |quiet: bool| {
        let mut command = Command::new(env!("CARGO_BIN_EXE_bloq"));
        command.args([
            "compile",
            "--gallery",
            "cnot",
            "-d",
            "3",
            "--backend",
            "ir-text",
            "--print",
        ]);
        if quiet {
            command.arg("--quiet");
        }
        command.output().expect("run CLI")
    };

    let normal = run(false);
    assert!(
        normal.status.success(),
        "{}",
        String::from_utf8_lossy(&normal.stderr)
    );
    let stderr = String::from_utf8_lossy(&normal.stderr);
    assert!(stderr.contains("d=3: Validating source"), "{stderr}");
    assert!(stderr.contains("d=3: Compilation complete"), "{stderr}");
    assert_finished_once(&stderr);
    assert!(
        !stderr.contains('\r'),
        "redirected progress must be line oriented"
    );

    let quiet = run(true);
    assert!(
        quiet.status.success(),
        "{}",
        String::from_utf8_lossy(&quiet.stderr)
    );
    assert!(
        quiet.stderr.is_empty(),
        "quiet compilation wrote to stderr: {}",
        String::from_utf8_lossy(&quiet.stderr)
    );
    for output in [&normal.stdout, &quiet.stdout] {
        assert!(
            output.starts_with(b"BLOQIR 1\n"),
            "stdout must start with IR"
        );
        assert!(
            !output
                .windows(b"Validating source".len())
                .any(|part| part == b"Validating source"),
            "progress leaked to stdout"
        );
    }
}

#[test]
fn elapsed_footer_covers_multi_target_file_run_but_not_errors() {
    let directory = tempfile::tempdir().expect("temporary outputs");
    let output = directory.path().join("compiled.bloqir");
    let run = Command::new(env!("CARGO_BIN_EXE_bloq"))
        .args([
            "compile",
            "--gallery",
            "cnot",
            "-d",
            "3,5",
            "--backend",
            "ir-text",
            "-o",
        ])
        .arg(&output)
        .output()
        .expect("compile two distances");
    assert!(
        run.status.success(),
        "{}",
        String::from_utf8_lossy(&run.stderr)
    );
    let stderr = String::from_utf8_lossy(&run.stderr);
    assert!(stderr.contains("d=3"), "{stderr}");
    assert!(stderr.contains("d=5"), "{stderr}");
    assert_finished_once(&stderr);
    assert!(directory.path().join("compiled-d3.bloqir").exists());
    assert!(directory.path().join("compiled-d5.bloqir").exists());

    let failed = Command::new(env!("CARGO_BIN_EXE_bloq"))
        .args(["compile", "-d", "3"])
        .arg(directory.path().join("missing.blog"))
        .output()
        .expect("run failing compile");
    assert!(!failed.status.success());
    assert!(
        !String::from_utf8_lossy(&failed.stderr).contains("Finished"),
        "failed compilation reported success"
    );

    let input = directory.path().join("t.blog");
    std::fs::write(&input, bloq_graph::GalleryItem::T.entry().blog()).unwrap();
    let rejected = Command::new(env!("CARGO_BIN_EXE_bloq"))
        .args(["compile", "-d", "3", "--quiet"])
        .arg(&input)
        .output()
        .expect("compile authored T without forced filling");
    assert!(!rejected.status.success());
    let stderr = String::from_utf8_lossy(&rejected.stderr);
    assert!(
        stderr.contains("unsupported Stim backend node: conditional component membership"),
        "{stderr}"
    );
    assert!(!stderr.contains("Finished"), "{stderr}");
    assert!(!directory.path().join("t-d3.stim").exists());
}

#[test]
fn compile_and_emit_print_exact_artifacts() {
    let directory = tempfile::tempdir().expect("temporary outputs");
    for (backend, extension) in [
        ("stim", "stim"),
        ("ir-text", "bloqir"),
        ("ir-binary", "bloq"),
    ] {
        let input = directory.path().join(format!("program.{extension}"));
        let saved = Command::new(env!("CARGO_BIN_EXE_bloq"))
            .args([
                "compile",
                "--gallery",
                "cnot",
                "-d",
                "3",
                "--quiet",
                "--backend",
                backend,
                "--output",
            ])
            .arg(&input)
            .output()
            .expect("compile saved artifact");
        assert!(
            saved.status.success(),
            "{}",
            String::from_utf8_lossy(&saved.stderr)
        );
        let expected = std::fs::read(&input).expect("read saved artifact");

        let printed = Command::new(env!("CARGO_BIN_EXE_bloq"))
            .args([
                "compile",
                "--gallery",
                "cnot",
                "-d",
                "3",
                "--quiet",
                "--backend",
                backend,
                "--print",
            ])
            .output()
            .expect("print compiled artifact");
        assert!(
            printed.status.success(),
            "{}",
            String::from_utf8_lossy(&printed.stderr)
        );
        assert_eq!(printed.stdout, expected);
        assert!(printed.stderr.is_empty());

        if backend != "stim" {
            let emitted = Command::new(env!("CARGO_BIN_EXE_bloq"))
                .arg("emit")
                .arg(&input)
                .args(["--quiet", "--backend", backend, "--print"])
                .output()
                .expect("print saved IR artifact");
            assert!(
                emitted.status.success(),
                "{}",
                String::from_utf8_lossy(&emitted.stderr)
            );
            assert_eq!(emitted.stdout, expected);
            assert!(emitted.stderr.is_empty());
        }
    }
}
