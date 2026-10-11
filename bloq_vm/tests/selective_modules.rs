#![cfg(test)]

use bloq_compile::{CompileConfig, CompileContext, SharedCompileCache};
use bloq_graph::{BlockGraph, PauliBasis, SelectiveKind};

#[test]
fn prepared_selective_modules_execute_both_arms_in_rotated_boundary_frames() {
    for kind in [SelectiveKind::XY, SelectiveKind::XZ, SelectiveKind::YZ] {
        for boundary in ["ZXZ", "XZX"] {
            let cache = SharedCompileCache::new();
            for choice in [false, true] {
                let invert = if choice { "!" } else { "" };
                let graph = BlockGraph::from_text(&format!(
                    "BLOG 1.0
module Cap {{
  in q: data = 0
  in basis
  out result = m
  0: Port [0,0,0] role=input
  1: {boundary} [0,0,1]
  2: {kind} [0,0,2]
  0 -> +Z
  1 -> +Z
  resolve 2 if basis
  m = measure 2
}}
module main {{
  in q: data = 0
  out control = {invert}source
  0: Port [0,0,0] role=input
  1: ZXZ [3,0,-1]
  cap: Cap @ [0,0,0] rotate Z 90
  0 -> cap.q
  source = measure 1
  control => cap.basis
}}"
                ))
                .unwrap();
                let compiler = CompileContext::with_shared_cache(CompileConfig::new(3), &cache);
                let prepared = compiler.compile_and_validate(&graph).unwrap().bloq;
                let report = bloq_vm::run_bloq_with_io(
                    &prepared,
                    4,
                    17,
                    |sim, context| {
                        let basis = if choice {
                            kind.pauli_if_true()
                        } else {
                            kind.pauli_if_false()
                        };
                        let input = context.inputs[0].qubit;
                        match basis {
                            PauliBasis::X => {}
                            PauliBasis::Y => sim.s(input),
                            PauliBasis::Z => sim.h(input),
                        }
                        Ok(())
                    },
                    |_, context| {
                        assert_eq!(context.branch_selectors, &[choice]);
                        Ok(())
                    },
                )
                .unwrap();
                assert_eq!(report.discarded, 0);
                assert!(report.detectors.iter().any(|d| d.per_shot.len() == 4));
                assert!(
                    report
                        .detectors
                        .iter()
                        .all(|d| d.per_shot.iter().all(|bit| !bit))
                );
                assert!(
                    report.observables.len() >= 2,
                    "both named measurements are retained"
                );
                assert!(
                    report
                        .observables
                        .iter()
                        .all(|observable| { observable.per_shot.iter().all(|&bit| !bit) }),
                    "{kind} {boundary} choice={choice}: {:?}",
                    report.observables
                );
            }
        }
    }
}
