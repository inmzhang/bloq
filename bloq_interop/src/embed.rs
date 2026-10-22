//! Direct surgery construction with explicit patch frames and disjoint lifetimes.

use std::collections::{BTreeMap, HashMap};

use bloq_graph::{
    Action, Basis, Block, BlockGraph, BlockKind, CubeKind, Direction, Expr, FeedbackTarget,
    MeasureTarget, PatchRotationKind, PauliBasis, Pipe, PortRole, SelectiveKind,
};
use glam::{IVec2, IVec3};

use crate::spider::{Instruction, Plan};

#[derive(Clone, Copy)]
struct Patch {
    end: IVec3,
    // A resource cap can finish one layer after the last data cube.
    ready: i32,
    rotated: bool,
}

impl Patch {
    fn memory(self) -> CubeKind {
        if self.rotated {
            CubeKind::ZXX
        } else {
            CubeKind::XZX
        }
    }
}

struct Builder {
    graph: BlockGraph,
    live: Vec<Option<Patch>>,
    actions: Vec<Action>,
    time: i32,
    corridor_ready: Vec<i32>,
    available: Vec<i32>,
    release_times: BTreeMap<usize, i32>,
    occupied_until: HashMap<IVec2, i32>,
    max_blocks: usize,
}

impl Builder {
    fn block(&mut self, pos: IVec3, kind: BlockKind) -> Result<(), String> {
        if self.graph.block_count() >= self.max_blocks {
            return Err(format!("block budget {} exhausted", self.max_blocks));
        }
        let block = Block::new(pos, kind);
        let offsets = block.reserved_offsets();
        self.graph.try_add_block(block).map_err(|e| e.to_string())?;
        for offset in offsets {
            let cell = pos + offset;
            self.occupied_until
                .entry(cell.truncate())
                .and_modify(|last| *last = (*last).max(cell.z))
                .or_insert(cell.z);
        }
        Ok(())
    }

    fn cube(&mut self, pos: IVec3, kind: CubeKind) -> Result<(), String> {
        self.block(pos, BlockKind::Cube(kind))
    }

    fn pipe(&mut self, from: IVec3, to: IVec3, hadamard: bool) -> Result<(), String> {
        let direction = Direction::try_from(to - from).map_err(|e| e.to_string())?;
        let pipe = Pipe::new(from, direction);
        self.graph
            .try_add_pipe(if hadamard { pipe.with_hadamard() } else { pipe })
            .map_err(|e| e.to_string())
    }

    fn port(&mut self, pos: IVec3, role: PortRole) -> Result<(), String> {
        self.block(pos, BlockKind::Port)?;
        self.graph
            .set_port_role(pos, role)
            .map_err(|e| e.to_string())
    }

    fn patch(&self, id: usize) -> Result<Patch, String> {
        self.live
            .get(id)
            .copied()
            .flatten()
            .ok_or_else(|| format!("patch {id} is not live"))
    }

    fn advance(&mut self, id: usize, time: i32) -> Result<Patch, String> {
        let mut patch = self.patch(id)?;
        while patch.end.z < time {
            let next = patch.end + IVec3::Z;
            self.cube(next, patch.memory())?;
            self.pipe(patch.end, next, false)?;
            patch.end = next;
        }
        patch.ready = patch.ready.max(patch.end.z);
        self.live[id] = Some(patch);
        Ok(patch)
    }

    fn after(time: i32, amount: i32) -> Result<i32, String> {
        time.checked_add(amount)
            .ok_or_else(|| "temporal coordinate overflow".into())
    }

    fn orient(&mut self, id: usize, rotated: bool) -> Result<(), String> {
        if self.patch(id)?.rotated == rotated {
            return Ok(());
        }
        let mut patch = self.advance(id, self.patch(id)?.ready)?;
        let time = Self::after(patch.end.z, 3)?;
        let start = patch.end + IVec3::Z;
        let movement = IVec2::new(0, if patch.end.y == 0 { 1 } else { -1 });
        let rotation =
            PatchRotationKind::new(if patch.rotated { Basis::Z } else { Basis::X }, movement)
                .map_err(|e| e.to_string())?;
        self.block(start, BlockKind::PatchRotation(rotation))?;
        self.pipe(patch.end, start, false)?;
        let end = rotation.end_position(start);
        patch.end = end + IVec3::Z;
        patch.ready = patch.end.z;
        patch.rotated = rotated;
        self.cube(patch.end, patch.memory())?;
        self.pipe(end, patch.end, false)?;
        self.live[id] = Some(patch);
        self.time = self.time.max(time);
        Ok(())
    }

    fn hadamard(&mut self, id: usize) -> Result<(), String> {
        let mut patch = self.patch(id)?;
        let previous = patch.end;
        patch.end.z = Self::after(patch.end.z, 1)?;
        patch.ready = patch.end.z;
        patch.rotated = !patch.rotated;
        self.cube(patch.end, patch.memory())?;
        self.pipe(previous, patch.end, true)?;
        self.live[id] = Some(patch);
        self.time = self.time.max(patch.end.z);
        Ok(())
    }

    fn cz(&mut self, control: usize, target: usize) -> Result<(), String> {
        if control == target {
            return Err("CZ requires distinct live patches".into());
        }
        self.orient(control, false)?;
        self.orient(target, true)?;
        let [start, end] = [self.patch(control)?.end.x + 1, self.patch(target)?.end.x]
            .map(|x| usize::try_from(x).expect("patch columns are non-negative"));
        let columns = start.min(end)..=start.max(end);
        // The corridor includes both vertical spokes; patch times reserve data endpoints.
        let corridor_ready = self.corridor_ready[columns.clone()]
            .iter()
            .copied()
            .max()
            .unwrap_or_default();
        let z = Self::after(
            self.patch(control)?
                .ready
                .max(self.patch(target)?.ready)
                .max(corridor_ready),
            1,
        )?;
        Self::after(z, 2)?;
        let control_id = control;
        let control = self.advance(control, z)?;
        let mut target_patch = self.advance(target, z)?;
        let target_junction = target_patch.end + IVec3::Z;
        self.cube(target_junction, CubeKind::XZZ)?;
        self.pipe(target_patch.end, target_junction, true)?;
        target_patch.end = target_junction + IVec3::Z;
        target_patch.ready = target_patch.end.z;
        self.cube(target_patch.end, target_patch.memory())?;
        self.pipe(target_junction, target_patch.end, true)?;
        self.live[target] = Some(target_patch);

        // Temporal bending transports the route frame without spatial H walls.
        let mut cursor = control.end + IVec3::X;
        self.cube(cursor, CubeKind::XZX)?;
        self.pipe(control.end, cursor, false)?;
        let upper = cursor + IVec3::Z;
        self.cube(upper, CubeKind::XZZ)?;
        self.pipe(cursor, upper, false)?;
        cursor = upper;
        while cursor.y > -1 {
            let next = cursor - IVec3::Y;
            self.cube(
                next,
                if next.y == -1 {
                    CubeKind::XXZ
                } else {
                    CubeKind::XZZ
                },
            )?;
            self.pipe(cursor, next, false)?;
            cursor = next;
        }
        let dx = (target_junction.x - cursor.x).signum();
        while cursor.x != target_junction.x {
            let next = cursor + IVec3::X * dx;
            self.cube(
                next,
                if next.x == target_junction.x {
                    CubeKind::XXZ
                } else {
                    CubeKind::ZXZ
                },
            )?;
            self.pipe(cursor, next, false)?;
            cursor = next;
        }
        while cursor.y < target_junction.y {
            let next = cursor + IVec3::Y;
            if next != target_junction {
                self.cube(next, CubeKind::XZZ)?;
            }
            self.pipe(cursor, next, false)?;
            cursor = next;
        }
        self.advance(control_id, target_patch.end.z)?;
        self.corridor_ready[columns].fill(target_patch.end.z);
        self.time = self.time.max(target_patch.end.z);
        Ok(())
    }

    fn z_feedback(&mut self, target: IVec3) {
        self.actions.push(Action::Feedback {
            targets: vec![FeedbackTarget {
                pauli: PauliBasis::Z,
                target,
                direction: Some(Direction::ZPLUS),
            }],
            condition: None,
        });
    }

    fn feedback(
        &mut self,
        measurement: usize,
        x_targets: &[usize],
        z_targets: &[usize],
    ) -> Result<(), String> {
        let time = Self::after(
            *self
                .release_times
                .get(&measurement)
                .ok_or("correction references an unavailable release")?,
            1,
        )?;
        // Bloq owns raw internal outcomes and their Pauli frames. Named readouts
        // are already corrected parities; only the causal dependency belongs here.
        for &id in x_targets.iter().chain(z_targets) {
            let patch = self.advance(id, time.max(self.patch(id)?.ready))?;
            self.time = self.time.max(patch.end.z);
        }
        Ok(())
    }

    fn phase(&mut self, id: usize, quarters: i64) -> Result<(), String> {
        let quarters = quarters.rem_euclid(8);
        if quarters == 0 {
            return Ok(());
        }
        if quarters % 2 == 1 {
            self.resource(id, true, quarters % 4 == 3)?;
        } else if quarters % 4 == 2 {
            self.resource(id, false, quarters == 6)?;
        }
        if matches!(quarters, 3..=5) {
            self.z_feedback(self.patch(id)?.end);
        }
        Ok(())
    }

    fn resource(&mut self, id: usize, t: bool, dagger: bool) -> Result<(), String> {
        let patch = self.patch(id)?;
        let direction = if patch.rotated {
            Direction::YPLUS
        } else {
            Direction::XPLUS
        };
        let column = (patch.end + direction.to_ivec3()).truncate();
        // Reserve the whole resource after prior geometry, including rotation footprints.
        let z = match self.occupied_until.get(&column) {
            Some(&last) => patch.ready.max(Self::after(last, if t { 2 } else { 1 })?),
            None => patch.ready,
        };
        Self::after(z, 1)?;
        let mut patch = self.advance(id, z)?;
        let resource = patch.end + direction.to_ivec3();
        self.cube(resource, patch.memory())?;
        self.pipe(patch.end, resource, false)?;
        let cap = resource + IVec3::Z;
        if t {
            let source = resource - IVec3::Z;
            self.block(source, BlockKind::T)?;
            self.pipe(source, resource, false)?;
            self.block(cap, BlockKind::Selective(SelectiveKind::XY))?;
            let name = format!("t{}", self.actions.len());
            self.actions.push(Action::Measure {
                target: MeasureTarget::Edge {
                    src: patch.end,
                    dir: direction,
                },
                name: name.clone(),
            });
            self.actions.push(Action::Resolve {
                target: cap,
                condition: if dagger {
                    Expr::Var(name)
                } else {
                    Expr::Not(Box::new(Expr::Var(name)))
                },
            });
        } else {
            self.block(cap, BlockKind::Y)?;
            if !dagger {
                self.z_feedback(patch.end);
            }
        }
        self.pipe(resource, cap, false)?;
        patch.ready = z + 1;
        self.live[id] = Some(patch);
        self.time = self.time.max(z + 1);
        Ok(())
    }
}

pub(crate) fn embed(
    plan: &Plan,
    max_blocks: usize,
) -> Result<(BlockGraph, Vec<IVec3>, Vec<IVec3>), String> {
    let mut b = Builder {
        graph: BlockGraph::new(),
        live: vec![None; plan.peak_patches],
        actions: Vec::new(),
        time: 0,
        corridor_ready: vec![
            0;
            plan.peak_patches
                .checked_mul(2)
                .ok_or("spatial coordinate overflow")?
        ],
        available: vec![0; plan.peak_patches],
        release_times: BTreeMap::new(),
        occupied_until: HashMap::default(),
        max_blocks,
    };
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();
    for instruction in &plan.instructions {
        match *instruction {
            Instruction::Prepare { patch, input, .. } => {
                if b.live.get(patch).is_none_or(Option::is_some) {
                    return Err("invalid patch preparation".into());
                }
                let x = i32::try_from(patch)
                    .ok()
                    .and_then(|x| x.checked_mul(2))
                    .ok_or("spatial coordinate overflow")?;
                let end = IVec3::new(x, 0, Builder::after(b.available[patch], 1)?);
                if let Some(input) = input {
                    let pos = end - IVec3::Z;
                    b.port(pos, PortRole::Input)?;
                    inputs.push((input, pos));
                }
                b.cube(end, CubeKind::XZX)?;
                if input.is_some() {
                    b.pipe(end - IVec3::Z, end, false)?;
                }
                b.live[patch] = Some(Patch {
                    end,
                    ready: end.z,
                    rotated: false,
                });
                b.time = b.time.max(end.z);
            }
            Instruction::Cz {
                patches: [a, c], ..
            } => b.cz(a, c)?,
            Instruction::Phase { patch, phase, .. } => {
                let r = phase.to_rational();
                if 4 % r.denom() != 0 {
                    return Err("only Clifford+T phases can be embedded".into());
                }
                b.phase(patch, r.numer() * (4 / r.denom()))?;
            }
            Instruction::Hadamard { patch } => b.hadamard(patch)?,
            Instruction::Release { patch, vertex } => {
                let live = b.patch(patch)?;
                let end = b
                    .advance(patch, Builder::after(live.end.z, 1)?.max(live.ready))?
                    .end;
                b.release_times.insert(vertex, end.z);
                b.live[patch] = None;
                b.available[patch] = Builder::after(end.z, 1)?;
                b.time = b.time.max(end.z);
            }
            Instruction::Feedback {
                measurement,
                ref x_targets,
                ref z_targets,
            } => b.feedback(measurement, x_targets, z_targets)?,
        }
    }
    let mut output_time = b.time;
    for &(_, id) in &plan.outputs {
        output_time = output_time.max(Builder::after(b.patch(id)?.end.z, 1)?);
    }
    for &(output, id) in &plan.outputs {
        let patch = b.advance(id, output_time - 1)?;
        let port = IVec3::new(patch.end.x, patch.end.y, output_time);
        b.port(port, PortRole::Output)?;
        b.pipe(patch.end, port, false)?;
        outputs.push((output, port));
    }
    for (index, wire) in plan.direct_wires.iter().enumerate() {
        let lane = plan
            .peak_patches
            .checked_add(index)
            .ok_or("spatial coordinate overflow")?;
        let x = i32::try_from(lane)
            .ok()
            .and_then(|x| x.checked_mul(2))
            .ok_or("spatial coordinate overflow")?;
        let input = IVec3::new(x, 0, 0);
        let middle = input + IVec3::Z;
        let output = middle + IVec3::Z;
        b.port(input, PortRole::Input)?;
        b.cube(middle, CubeKind::XZX)?;
        b.port(output, PortRole::Output)?;
        b.pipe(input, middle, false)?;
        b.pipe(middle, output, wire.hadamard)?;
        inputs.push((wire.input, input));
        outputs.push((wire.output, output));
    }
    inputs.sort_by_key(|&(i, _)| i);
    outputs.sort_by_key(|&(i, _)| i);
    b.graph
        .set_actions_deferred(b.actions)
        .map_err(|e| e.to_string())?;
    b.graph.validate_source().map_err(|e| e.to_string())?;
    Ok((
        b.graph,
        inputs.into_iter().map(|(_, p)| p).collect(),
        outputs.into_iter().map(|(_, p)| p).collect(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use bloq_graph::verify::{BoundaryOrder, LogicalVerifier, QuizxGraph};
    use quizx::circuit::Circuit;
    use quizx::vec_graph::Graph;

    #[test]
    fn free_resource_outlets_reuse_data_endpoints_in_both_frames() {
        for (count, blocks, output_time) in [(1, 6, 2), (2, 10, 3), (3, 15, 5)] {
            let mut plan = Plan {
                peak_patches: 1,
                outputs: vec![(0, 0)],
                instructions: vec![Instruction::Prepare {
                    patch: 0,
                    vertex: 0,
                    input: Some(0),
                }],
                ..Plan::default()
            };
            let mut expected = Circuit::new(1);
            for index in 0..count {
                plan.instructions.push(Instruction::Phase {
                    patch: 0,
                    vertex: index,
                    phase: (1, 4).into(),
                });
                expected.add_gate("t", vec![0]);
                if index + 1 < count {
                    plan.instructions.push(Instruction::Hadamard { patch: 0 });
                    expected.add_gate("h", vec![0]);
                }
            }
            let (graph, inputs, outputs) = embed(&plan, 1000).unwrap();
            assert_eq!(graph.block_count(), blocks);
            assert_eq!(outputs[0].z, output_time);
            LogicalVerifier::with_boundaries_and_internal_measurements(
                &graph,
                BoundaryOrder::new(inputs, outputs),
            )
            .unwrap()
            .verify(&expected.to_graph(), 8, 29)
            .unwrap();
            bloq_compile::CompileContext::new(bloq_compile::CompileConfig::new(3))
                .compile_and_validate(&graph)
                .unwrap();
        }
    }

    #[test]
    fn disjoint_cz_routes_overlap_but_crossing_or_shared_patch_routes_wait() {
        for (pairs, route_layers) in [
            ([[0, 1], [2, 3]], 1),
            ([[1, 0], [3, 2]], 1),
            ([[0, 2], [1, 3]], 2),
            ([[0, 1], [1, 2]], 2),
        ] {
            let mut plan = Plan {
                peak_patches: 4,
                outputs: (0..4).map(|i| (i, i)).collect(),
                instructions: (0..4)
                    .map(|patch| Instruction::Prepare {
                        patch,
                        vertex: patch,
                        input: Some(patch),
                    })
                    .collect(),
                ..Plan::default()
            };
            let mut expected = Circuit::new(4);
            for patches in pairs {
                plan.instructions.push(Instruction::Cz {
                    patches,
                    vertices: patches,
                });
                expected.add_gate("cz", patches.to_vec());
            }
            let (graph, inputs, outputs) = embed(&plan, 1000).unwrap();
            let occupied_layers: std::collections::BTreeSet<_> = graph
                .pipes()
                .filter(|p| p.src().y == -1 && p.dst().y == -1)
                .map(|p| p.src().z)
                .collect();
            assert_eq!(occupied_layers.len(), route_layers, "{pairs:?}");
            LogicalVerifier::with_boundaries_and_internal_measurements(
                &graph,
                BoundaryOrder::new(inputs, outputs),
            )
            .unwrap()
            .verify(&expected.to_graph(), 8, 23)
            .unwrap();
            bloq_compile::CompileContext::new(bloq_compile::CompileConfig::new(3))
                .compile_and_validate(&graph)
                .unwrap();
        }
    }

    #[test]
    fn teleportation_uses_native_output_frames() {
        let plan = Plan {
            peak_patches: 2,
            outputs: vec![(0, 1)],
            instructions: vec![
                Instruction::Prepare {
                    patch: 0,
                    vertex: 0,
                    input: Some(0),
                },
                Instruction::Prepare {
                    patch: 1,
                    vertex: 1,
                    input: None,
                },
                Instruction::Cz {
                    patches: [0, 1],
                    vertices: [0, 1],
                },
                Instruction::Release {
                    patch: 0,
                    vertex: 0,
                },
                Instruction::Feedback {
                    measurement: 0,
                    x_targets: vec![1],
                    z_targets: vec![],
                },
            ],
            ..Plan::default()
        };
        let (graph, inputs, outputs) = embed(&plan, 1000).unwrap();
        let expected: QuizxGraph =
            Circuit::from_qasm("OPENQASM 2.0; include \"qelib1.inc\"; qreg q[1]; h q[0];")
                .unwrap()
                .to_graph();
        let verifier = LogicalVerifier::with_boundaries_and_internal_measurements(
            &graph,
            BoundaryOrder::new(inputs, outputs),
        )
        .unwrap_or_else(|e| panic!("{e}\n{}", graph.to_blog_body_text()));
        verifier.verify(&expected, 8, 17).unwrap();
        bloq_compile::CompileContext::new(bloq_compile::CompileConfig::new(3))
            .compile_and_validate(&graph)
            .unwrap();
    }

    #[test]
    fn independent_resources_overlap_and_repeated_sources_have_separate_lifetimes() {
        let mut plan = Plan {
            peak_patches: 2,
            outputs: vec![(0, 0), (1, 1)],
            ..Plan::default()
        };
        for patch in 0..2 {
            plan.instructions.push(Instruction::Prepare {
                patch,
                vertex: patch,
                input: Some(patch),
            });
        }
        for patch in [0, 1, 0] {
            plan.instructions.push(Instruction::Phase {
                patch,
                vertex: patch,
                phase: (1, 4).into(),
            });
        }
        let (graph, inputs, outputs) = embed(&plan, 100_000).unwrap();
        let first_sources: Vec<_> = graph
            .blocks()
            .filter(|b| b.kind() == BlockKind::T)
            .map(Block::pos)
            .collect();
        let earliest = first_sources.iter().map(|p| p.z).min().unwrap();
        assert_eq!(first_sources.iter().filter(|p| p.z == earliest).count(), 2);
        assert!(
            embed(&plan, graph.block_count() - 1)
                .unwrap_err()
                .contains("budget")
        );
        let expected: QuizxGraph =
            Circuit::from_qasm("OPENQASM 2.0; include \"qelib1.inc\"; qreg q[2]; s q[0]; t q[1];")
                .unwrap()
                .to_graph();
        LogicalVerifier::with_boundaries_and_internal_measurements(
            &graph,
            BoundaryOrder::new(inputs, outputs),
        )
        .unwrap()
        .verify(&expected, 8, 17)
        .unwrap();
        bloq_compile::CompileContext::new(bloq_compile::CompileConfig::new(3))
            .compile_and_validate(&graph)
            .unwrap();
    }

    #[test]
    fn direct_geometry_preserves_signed_phases_and_routed_interactions() {
        for (qubits, gates) in [
            (1, "h q[0];"),
            (1, "h q[0]; t q[0];"),
            (1, "h q[0]; tdg q[0];"),
            (1, "t q[0];"),
            (1, "tdg q[0];"),
            (1, "t q[0]; s q[0];"),
            (1, "tdg q[0]; sdg q[0];"),
            (1, "s q[0];"),
            (1, "sdg q[0];"),
            (1, "z q[0];"),
            (2, "cz q[0],q[1];"),
            (2, "t q[0]; cz q[0],q[1];"),
            (2, "t q[1]; cz q[0],q[1];"),
            (2, "h q[0]; t q[0]; cz q[0],q[1]; tdg q[1];"),
            (2, "swap q[0],q[1]; h q[0];"),
            (2, "h q[0]; cx q[0],q[1]; h q[1]; cx q[1],q[0];"),
            (
                2,
                "h q[0]; cx q[0],q[1]; t q[1]; h q[1]; cx q[1],q[0]; tdg q[0];",
            ),
            (3, "cz q[0],q[2]; cz q[1],q[0]; cz q[2],q[1];"),
        ] {
            let circuit = Circuit::from_qasm(&format!(
                "OPENQASM 2.0; include \"qelib1.inc\"; qreg q[{qubits}]; {gates}"
            ))
            .unwrap();
            let circuit = crate::source::normalize(&circuit).unwrap();
            let expected: QuizxGraph = circuit.to_graph();
            let mut zx: Graph = circuit.to_graph();
            quizx::simplify::full_simp(&mut zx);
            let plan = crate::spider::plan(&zx).unwrap();
            let (graph, inputs, outputs) =
                embed(&plan, 100_000).unwrap_or_else(|e| panic!("{gates}: {e}"));
            assert!(
                graph
                    .pipes()
                    .all(|p| !p.is_hadamard() || !p.dir().is_spatial())
            );
            let verifier = LogicalVerifier::with_boundaries_and_internal_measurements(
                &graph,
                BoundaryOrder::new(inputs, outputs),
            )
            .unwrap_or_else(|e| panic!("{gates}: {e}\n{}", graph.to_blog_body_text()));
            bloq_compile::CompileContext::new(bloq_compile::CompileConfig::new(3))
                .compile_and_validate(&graph)
                .unwrap_or_else(|e| panic!("{gates}: {e}"));
            verifier
                .verify(&expected, 8, 1042)
                .unwrap_or_else(|e| panic!("{gates}: {e}\n{}", graph.to_blog_body_text()));
        }
    }
}
