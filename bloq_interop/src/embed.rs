//! Direct surgery construction with explicit patch frames and disjoint lifetimes.

use std::collections::{BTreeMap, HashMap};

use bloq_graph::{
    Action, Block, BlockGraph, BlockKind, CubeHeight, CubeKind, Direction, Expr, FeedbackTarget,
    MAX_CUBE_HEIGHT_CELLS, MeasureTarget, PauliBasis, Pipe, PortRole, SelectiveKind,
};
use glam::{IVec2, IVec3};

use crate::spider::{Instruction, Plan};
use crate::{GridOptions, LayoutOptions, grid::Grid};

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum TPlacement {
    Fixed,
    Available,
}

#[derive(Clone, Copy)]
struct Patch {
    slot: usize,
    end: IVec3,
    // A resource cap can finish one layer after the last data cube.
    ready: i32,
    hadamard: bool,
}

impl Patch {
    fn memory(self) -> CubeKind {
        if self.hadamard {
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
    available: BTreeMap<usize, i32>,
    release_times: BTreeMap<usize, i32>,
    occupied_until: HashMap<IVec2, i32>,
    max_blocks: usize,
    max_temporal_extent: Option<u32>,
    grid: Option<Grid>,
    t_placement: TPlacement,
}

impl Builder {
    fn block(&mut self, block: Block) -> Result<(), String> {
        if self.graph.block_count() >= self.max_blocks {
            return Err(format!("block budget {} exhausted", self.max_blocks));
        }
        let pos = block.pos();
        let offsets = block.reserved_offsets();
        if let Some(limit) = self.max_temporal_extent
            && offsets
                .iter()
                .any(|offset| i64::from(pos.z) + i64::from(offset.z) >= i64::from(limit))
        {
            return Err(format!("temporal extent limit {limit} exhausted"));
        }
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
        self.block(Block::new(pos, BlockKind::Cube(kind)))
    }

    fn pipe(&mut self, from: IVec3, to: IVec3, hadamard: bool) -> Result<(), String> {
        let direction = Direction::try_from(to - from).map_err(|e| e.to_string())?;
        let pipe = Pipe::new(from, direction);
        self.graph
            .try_add_pipe(if hadamard { pipe.with_hadamard() } else { pipe })
            .map_err(|e| e.to_string())
    }

    fn port(&mut self, pos: IVec3, role: PortRole) -> Result<(), String> {
        self.block(
            Block::new(pos, BlockKind::Port)
                .with_port_role(role)
                .map_err(|e| e.to_string())?,
        )
    }

    fn patch(&self, id: usize) -> Result<Patch, String> {
        self.live
            .get(id)
            .copied()
            .flatten()
            .ok_or_else(|| format!("patch {id} is not live"))
    }

    fn home(&self, slot: usize) -> Result<IVec2, String> {
        if let Some(grid) = &self.grid {
            return grid.home(slot);
        }
        let x = i32::try_from(slot)
            .ok()
            .and_then(|x| x.checked_mul(2))
            .ok_or("spatial coordinate overflow")?;
        Ok(IVec2::new(x, 0))
    }

    fn advance(&mut self, id: usize, time: i32) -> Result<Patch, String> {
        let mut patch = self.patch(id)?;
        while patch.end.z < time {
            let next = patch.end + IVec3::Z;
            // Keep a unit endpoint for future junctions; tall memory preserves every cell and round.
            let cells = ((time - patch.end.z - 1).max(1) as u32).min(MAX_CUBE_HEIGHT_CELLS);
            let height = CubeHeight::new(cells, 1, 0).map_err(|e| e.to_string())?;
            self.block(
                Block::new(next, BlockKind::Cube(patch.memory()))
                    .with_height(height)
                    .map_err(|e| e.to_string())?,
            )?;
            self.pipe(patch.end, next, false)?;
            patch.end.z += cells as i32;
        }
        patch.ready = patch.ready.max(patch.end.z);
        self.live[id] = Some(patch);
        Ok(patch)
    }

    fn after(time: i32, amount: i32) -> Result<i32, String> {
        time.checked_add(amount)
            .ok_or_else(|| "temporal coordinate overflow".into())
    }

    fn hadamard(&mut self, id: usize) -> Result<(), String> {
        let mut patch = self.patch(id)?;
        let previous = patch.end;
        patch.end.z = Self::after(patch.end.z, 1)?;
        patch.ready = patch.end.z;
        patch.hadamard = !patch.hadamard;
        self.cube(patch.end, patch.memory())?;
        self.pipe(previous, patch.end, true)?;
        self.live[id] = Some(patch);
        self.time = self.time.max(patch.end.z);
        Ok(())
    }

    fn route(&self, control: Patch, target: Patch) -> Result<(Vec<IVec2>, i32), String> {
        let source = control.end.truncate();
        let from = source
            + if control.hadamard {
                IVec2::NEG_Y
            } else {
                IVec2::X
            };
        let to = target.end.truncate();
        let mut path = vec![from];
        let exit = if control.hadamard {
            from + IVec2::new(if to.x < source.x { -1 } else { 1 }, 0)
        } else {
            from - IVec2::Y
        };
        path.push(exit);
        // The +X target arm stays clear of a neighbouring source in pitch-two rows.
        let mut tail = if target.hadamard {
            vec![to - IVec2::Y]
        } else {
            vec![to + IVec2::X - IVec2::Y, to + IVec2::X]
        };
        tail.push(to);
        let entry = tail[0];
        let mut z = Self::after(control.ready.max(target.ready), 1)?;
        if let Some(grid) = &self.grid {
            for cell in path.iter().chain(&tail) {
                if let Some(&last) = self.occupied_until.get(cell) {
                    z = z.max(Self::after(last, 1)?);
                }
            }
            let stub =
                |p: IVec2| p != exit && p != entry && (path.contains(&p) || tail.contains(&p));
            // Try a concurrent detour before waiting for the shortest static route.
            let middle = match grid.path(exit, entry, |p| {
                stub(p) || self.occupied_until.get(&p).is_some_and(|&last| last >= z)
            })? {
                Some(path) => path,
                None => grid
                    .path(exit, entry, stub)?
                    .ok_or("no route inside the grid canvas")?,
            };
            path.extend(middle.into_iter().skip(1));
            path.extend(tail.into_iter().skip(1));
        } else {
            let dx = (entry.x - exit.x).signum();
            while path.last().expect("route has a start").x != entry.x {
                path.push(*path.last().expect("route has a start") + IVec2::X * dx);
            }
            path.extend(tail.into_iter().skip(1));
        }
        for cell in &path {
            if let Some(&last) = self.occupied_until.get(cell) {
                z = z.max(Self::after(last, 1)?);
            }
        }
        Ok((path, z))
    }

    fn cz(&mut self, control: usize, target: usize) -> Result<(), String> {
        if control == target {
            return Err("CZ requires distinct live patches".into());
        }
        let (path, z) = self.route(self.patch(control)?, self.patch(target)?)?;
        Self::after(z, 2)?;
        let control_id = control;
        let control = self.advance(control, z)?;
        let mut target_patch = self.advance(target, z)?;
        let target_junction = target_patch.end + IVec3::Z;
        self.cube(target_junction, target_patch.memory())?;
        self.pipe(target_patch.end, target_junction, false)?;
        target_patch.end = target_junction + IVec3::Z;
        target_patch.ready = target_patch.end.z;
        self.cube(target_patch.end, target_patch.memory())?;
        self.pipe(target_junction, target_patch.end, false)?;
        self.live[target] = Some(target_patch);

        let mut cursor = path[0].extend(z);
        self.cube(cursor, control.memory())?;
        self.pipe(control.end, cursor, false)?;
        for (index, &cell) in path.iter().enumerate() {
            let next = cell.extend(z + 1);
            if next != target_junction {
                let kind = if index == 0 {
                    if control.hadamard {
                        CubeKind::ZXZ
                    } else {
                        CubeKind::XZZ
                    }
                } else {
                    let before = cell - path[index - 1];
                    let after = path[index + 1] - cell;
                    if before.x != 0 && after.x != 0 {
                        CubeKind::ZXZ
                    } else if before.y != 0 && after.y != 0 {
                        CubeKind::XZZ
                    } else {
                        CubeKind::XXZ
                    }
                };
                self.cube(next, kind)?;
            }
            // A colour change moves the interaction H onto the target's temporal legs.
            self.pipe(cursor, next, next == target_junction)?;
            cursor = next;
        }
        self.advance(control_id, target_patch.end.z)?;
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
        let mut direction = if patch.hadamard {
            Direction::YPLUS
        } else {
            Direction::XPLUS
        };
        let availability = |direction: Direction| -> Result<_, String> {
            let column = (patch.end + direction.to_ivec3()).truncate();
            let last = self.occupied_until.get(&column);
            // Reserve source, junction and cap after all prior routing geometry.
            let z = match last {
                Some(&last) => patch.ready.max(Self::after(last, if t { 2 } else { 1 })?),
                None => patch.ready,
            };
            Ok((z, last.is_none()))
        };
        let mut cost = availability(direction)?;
        if t && self.grid.is_some() && self.t_placement == TPlacement::Available {
            let opposite = availability(direction.negate())?;
            if opposite < cost {
                direction = direction.negate();
                cost = opposite;
            }
        }
        let z = cost.0;
        Self::after(z, 1)?;
        let mut patch = self.advance(id, z)?;
        let resource = patch.end + direction.to_ivec3();
        self.cube(resource, patch.memory())?;
        self.pipe(patch.end, resource, false)?;
        let cap = resource + IVec3::Z;
        if t {
            let source = resource - IVec3::Z;
            self.block(Block::new(source, BlockKind::T))?;
            self.pipe(source, resource, false)?;
            self.block(Block::new(cap, BlockKind::Selective(SelectiveKind::XY)))?;
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
            self.block(Block::new(cap, BlockKind::Y))?;
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

pub(crate) struct Embedding {
    pub graph: BlockGraph,
    pub input_ports: Vec<IVec3>,
    pub output_ports: Vec<IVec3>,
    pub grid: Option<GridOptions>,
    pub patch_locations: Vec<IVec2>,
    pub routing_bounds: Option<[IVec2; 2]>,
}

pub(crate) fn embed(
    plan: &Plan,
    options: &LayoutOptions,
    t_placement: TPlacement,
) -> Result<Embedding, String> {
    let extra_patches = options.extra_patches;
    // More locations than preparations cannot be used, even with an unbounded request.
    let preparations = plan
        .instructions
        .iter()
        .filter(|i| matches!(i, Instruction::Prepare { .. }))
        .count();
    let slots =
        plan.peak_patches + extra_patches.min(preparations.saturating_sub(plan.peak_patches));
    let locations = slots
        .checked_add(plan.direct_wires.len())
        .ok_or("patch count overflow")?;
    let grid = options
        .grid
        .map(|options| Grid::new(options, locations))
        .transpose()?;
    let mut b = Builder {
        graph: BlockGraph::new(),
        live: vec![None; plan.peak_patches],
        actions: Vec::new(),
        time: 0,
        available: (0..slots).map(|slot| (slot, 0)).collect(),
        release_times: BTreeMap::new(),
        occupied_until: HashMap::default(),
        max_blocks: options.max_blocks,
        max_temporal_extent: options.max_temporal_extent,
        grid,
        t_placement,
    };
    let patch_locations = (0..locations)
        .map(|slot| b.home(slot))
        .collect::<Result<Vec<_>, _>>()?;
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();
    let mut used_slots = 0;
    for instruction in &plan.instructions {
        match *instruction {
            Instruction::Prepare { patch, input, .. } => {
                if b.live.get(patch).is_none_or(Option::is_some) {
                    return Err("invalid patch preparation".into());
                }
                let slot = if extra_patches == 0 {
                    patch
                } else {
                    *b.available
                        .iter()
                        .min_by_key(|&(slot, ready)| (*ready, *slot))
                        .ok_or("no free patch location")?
                        .0
                };
                let ready = b.available.remove(&slot).ok_or("patch location is live")?;
                used_slots = used_slots.max(slot + 1);
                let end = b.home(slot)?.extend(Builder::after(ready, 1)?);
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
                    slot,
                    end,
                    ready: end.z,
                    hadamard: false,
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
                b.available.insert(live.slot, Builder::after(end.z, 1)?);
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
        let lane = used_slots
            .checked_add(index)
            .ok_or("spatial coordinate overflow")?;
        let input = b.home(lane)?.extend(0);
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
    b.graph = crate::hadamard::optimize(b.graph)?;
    b.graph.validate_source().map_err(|e| e.to_string())?;
    Ok(Embedding {
        graph: b.graph,
        input_ports: inputs.into_iter().map(|(_, p)| p).collect(),
        output_ports: outputs.into_iter().map(|(_, p)| p).collect(),
        grid: b.grid.as_ref().map(Grid::options),
        patch_locations,
        routing_bounds: b.grid.as_ref().map(Grid::bounds),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use bloq_graph::verify::{BoundaryOrder, LogicalVerifier, QuizxGraph};
    use quizx::circuit::Circuit;
    use quizx::vec_graph::Graph;

    fn embed(
        plan: &Plan,
        max_blocks: usize,
        extra_patches: usize,
    ) -> Result<(BlockGraph, Vec<IVec3>, Vec<IVec3>), String> {
        super::embed(
            plan,
            &LayoutOptions {
                max_blocks,
                extra_patches,
                grid: None,
                ..Default::default()
            },
            TPlacement::Fixed,
        )
        .map(|e| (e.graph, e.input_ports, e.output_ports))
    }

    #[test]
    fn alternate_t_faces_overlap_resource_lifetimes_without_changing_the_map() {
        let mut plan = Plan {
            peak_patches: 2,
            outputs: vec![(0, 0), (1, 1)],
            direct_wires: vec![crate::spider::DirectWire {
                input: 2,
                output: 2,
                hadamard: false,
            }],
            instructions: (0..2)
                .map(|patch| Instruction::Prepare {
                    patch,
                    vertex: patch,
                    input: Some(patch),
                })
                .collect(),
        };
        let mut expected = Circuit::new(3);
        plan.instructions.push(Instruction::Cz {
            patches: [0, 1],
            vertices: [0, 1],
        });
        expected.add_gate("cz", vec![0, 1]);
        for phase in [0, 1] {
            for (gate, angle) in [("t", 1), ("t", 1), ("tdg", -1)] {
                plan.instructions.push(Instruction::Phase {
                    patch: 1,
                    vertex: 1,
                    phase: (angle, 4).into(),
                });
                expected.add_gate(gate, vec![1]);
            }
            if phase == 0 {
                plan.instructions.push(Instruction::Hadamard { patch: 1 });
                expected.add_gate("h", vec![1]);
            }
        }
        // Follow the changed Hadamard frame after both X-face resource lifetimes.
        plan.instructions.push(Instruction::Cz {
            patches: [0, 1],
            vertices: [0, 1],
        });
        expected.add_gate("cz", vec![0, 1]);
        for (gate, angle) in [("t", 1), ("t", 1), ("tdg", -1)] {
            plan.instructions.push(Instruction::Phase {
                patch: 1,
                vertex: 1,
                phase: (angle, 4).into(),
            });
            expected.add_gate(gate, vec![1]);
        }
        let options = LayoutOptions {
            grid: Some(crate::GridOptions {
                columns: 1,
                ..Default::default()
            }),
            ..Default::default()
        };
        let times = [TPlacement::Fixed, TPlacement::Available].map(|placement| {
            let Embedding {
                graph,
                input_ports: inputs,
                output_ports: outputs,
                ..
            } = super::embed(&plan, &options, placement).unwrap();
            assert_eq!(
                graph.blocks().filter(|b| b.kind() == BlockKind::T).count(),
                9
            );
            if placement == TPlacement::Available {
                let directions: std::collections::HashSet<_> = graph
                    .actions()
                    .into_iter()
                    .filter_map(|action| match action {
                        Action::Measure {
                            target: MeasureTarget::Edge { dir, .. },
                            ..
                        } => Some(dir),
                        _ => None,
                    })
                    .collect();
                assert!(
                    [Direction::XMINUS, Direction::YMINUS]
                        .iter()
                        .all(|direction| directions.contains(direction))
                );
            }
            LogicalVerifier::with_boundaries_and_internal_measurements(
                &graph,
                BoundaryOrder::new(inputs, outputs.clone()),
            )
            .unwrap()
            .verify(&expected.to_graph(), 16, 2903)
            .unwrap();
            for proxies in [false, true] {
                bloq_compile::CompileContext::new(
                    bloq_compile::CompileConfig::new(3).with_prepare_t_with_mpps(proxies),
                )
                .compile_and_validate(&graph)
                .unwrap();
            }
            outputs[0].z
        });
        assert!(
            times[1] < times[0],
            "fixed {}, alternate {}",
            times[0],
            times[1]
        );
    }

    #[test]
    fn fixed_home_cz_preserves_all_frames_and_signed_followers() {
        for frames in [[false, false], [false, true], [true, false], [true, true]] {
            for reversed in [false, true] {
                let pair = if reversed { [1, 0] } else { [0, 1] };
                let delayed = pair[0];
                let mut plan = Plan {
                    peak_patches: 2,
                    outputs: vec![(0, 0), (1, 1)],
                    instructions: (0..2)
                        .map(|patch| Instruction::Prepare {
                            patch,
                            vertex: patch,
                            input: Some(patch),
                        })
                        .collect(),
                    ..Plan::default()
                };
                let mut expected = Circuit::new(2);
                for (patch, hadamard) in frames.into_iter().enumerate() {
                    if hadamard {
                        plan.instructions.push(Instruction::Hadamard { patch });
                        expected.add_gate("h", vec![patch]);
                    }
                }
                for _ in 0..2 {
                    plan.instructions.push(Instruction::Phase {
                        patch: delayed,
                        vertex: delayed,
                        phase: (1, 4).into(),
                    });
                    expected.add_gate("t", vec![delayed]);
                }
                plan.instructions.push(Instruction::Cz {
                    patches: pair,
                    vertices: pair,
                });
                expected.add_gate("cz", pair.to_vec());
                for (patch, quarter, gate) in [(0, -1, "tdg"), (1, 1, "t")] {
                    plan.instructions.extend([
                        Instruction::Hadamard { patch },
                        Instruction::Phase {
                            patch,
                            vertex: patch,
                            phase: (quarter, 4).into(),
                        },
                    ]);
                    expected.add_gate("h", vec![patch]);
                    expected.add_gate(gate, vec![patch]);
                }
                plan.instructions.push(Instruction::Cz {
                    patches: pair,
                    vertices: pair,
                });
                expected.add_gate("cz", pair.to_vec());
                for placement in [TPlacement::Fixed, TPlacement::Available] {
                    for grid in [
                        None,
                        Some(crate::GridOptions::default()),
                        Some(crate::GridOptions {
                            columns: 1,
                            ..Default::default()
                        }),
                    ] {
                        let options = LayoutOptions {
                            grid,
                            ..Default::default()
                        };
                        let Embedding {
                            graph,
                            input_ports,
                            output_ports,
                            ..
                        } = super::embed(&plan, &options, placement).unwrap();
                        assert!(
                            graph
                                .blocks()
                                .all(|block| !matches!(block.kind(), BlockKind::PatchRotation(_)))
                        );
                        assert!(
                            graph
                                .pipes()
                                .all(|pipe| !pipe.is_hadamard() || !pipe.dir().is_spatial())
                        );
                        LogicalVerifier::with_boundaries_and_internal_measurements(
                            &graph,
                            BoundaryOrder::new(input_ports, output_ports),
                        )
                        .unwrap()
                        .verify(&expected.to_graph(), 8, 0xC2EA)
                        .unwrap();
                    }
                }
            }
        }
    }

    #[test]
    fn extra_locations_overlap_independent_lifetimes_and_keep_passthroughs_clear() {
        let mut plan = Plan {
            peak_patches: 3,
            outputs: vec![(0, 1), (1, 2)],
            direct_wires: vec![crate::spider::DirectWire {
                input: 2,
                output: 2,
                hadamard: false,
            }],
            ..Plan::default()
        };
        for (input, target) in [(0, 1), (1, 2)] {
            let vertex = input * 2;
            plan.instructions.extend([
                Instruction::Prepare {
                    patch: 0,
                    vertex,
                    input: Some(input),
                },
                Instruction::Prepare {
                    patch: target,
                    vertex: vertex + 1,
                    input: None,
                },
                Instruction::Cz {
                    patches: [0, target],
                    vertices: [vertex, vertex + 1],
                },
                Instruction::Release { patch: 0, vertex },
                Instruction::Feedback {
                    measurement: vertex,
                    x_targets: vec![target],
                    z_targets: vec![],
                },
            ]);
        }
        let expected: QuizxGraph = Circuit::from_qasm("OPENQASM 2.0; qreg q[3]; h q[0]; h q[1];")
            .unwrap()
            .to_graph();
        for grid in [
            None,
            Some(crate::GridOptions {
                columns: 1,
                ..Default::default()
            }),
        ] {
            let metrics = [0, 1, usize::MAX].map(|extra| {
                let Embedding {
                    graph,
                    input_ports: inputs,
                    output_ports: outputs,
                    patch_locations,
                    ..
                } = super::embed(
                    &plan,
                    &LayoutOptions {
                        grid,
                        extra_patches: extra,
                        ..Default::default()
                    },
                    TPlacement::Fixed,
                )
                .unwrap();
                assert_eq!(patch_locations.len(), if extra == 0 { 4 } else { 5 });
                LogicalVerifier::with_boundaries_and_internal_measurements(
                    &graph,
                    BoundaryOrder::new(inputs, outputs),
                )
                .unwrap()
                .verify(&expected, 8, 29)
                .unwrap();
                bloq_compile::CompileContext::new(bloq_compile::CompileConfig::new(3))
                    .compile_and_validate(&graph)
                    .unwrap();
                let (x, _, z) = graph.spans().unwrap();
                (*x.end() - *x.start(), *z.end() - *z.start())
            });
            if grid.is_none() {
                assert!(metrics[1].0 > metrics[0].0);
            }
            assert!(metrics[1].1 < metrics[0].1);
            assert_eq!(metrics[1], metrics[2]);
        }
    }

    #[test]
    fn tall_idle_memory_preserves_occupied_cells_and_physical_rounds() {
        for rotated in [false, true] {
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
            if rotated {
                plan.instructions.push(Instruction::Hadamard { patch: 1 });
            }
            plan.instructions
                .extend((0..7).map(|_| Instruction::Hadamard { patch: 0 }));
            let (graph, inputs, outputs) = embed(&plan, 100, 0).unwrap();
            assert_eq!(graph.occupied_positions().count(), 20);
            assert!(graph.block_count() < 20);
            assert_eq!(outputs.iter().map(|p| p.z).collect::<Vec<_>>(), [9, 9]);
            embed(&plan, graph.block_count() - 1, 0).unwrap_err();

            // Expand only idle memory to the previous unit-cube construction.
            let mut reference = BlockGraph::new();
            for block in graph.blocks() {
                for z in 0..block.height_cells() as i32 {
                    let pos = block.pos() + IVec3::Z * z;
                    reference
                        .try_add_block(Block::new(pos, block.kind()))
                        .unwrap();
                    if z != 0 {
                        reference
                            .try_add_pipe(Pipe::new(pos - IVec3::Z, Direction::ZPLUS))
                            .unwrap();
                    }
                }
            }
            for pipe in graph.pipes() {
                reference.try_add_pipe(pipe.clone()).unwrap();
            }
            let mut expected = Circuit::new(2);
            expected.add_gate("h", vec![0]);
            if rotated {
                expected.add_gate("h", vec![1]);
            }
            LogicalVerifier::with_boundaries_and_internal_measurements(
                &graph,
                BoundaryOrder::new(inputs, outputs),
            )
            .unwrap()
            .verify(&expected.to_graph(), 8, 41)
            .unwrap();
            for distance in [3, 5, 25] {
                let compile = |source| {
                    let mut program = bloq_compile::CompileContext::new(
                        bloq_compile::CompileConfig::new(distance),
                    )
                    .compile_and_validate(source)
                    .unwrap()
                    .bloq;
                    program.flatten().unwrap();
                    program
                };
                let (actual, unit) = (compile(&graph), compile(&reference));
                assert_eq!(actual.qubit_count().unwrap(), unit.qubit_count().unwrap());
                assert_eq!(actual.measurement_count(), unit.measurement_count());
            }
        }
    }

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
            let (graph, inputs, outputs) = embed(&plan, 1000, 0).unwrap();
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
        for (pairs, route_layers, hadamard) in [
            ([[0, 1], [2, 3]], 1, None),
            ([[1, 0], [3, 2]], 1, None),
            ([[0, 2], [1, 3]], 2, None),
            ([[0, 1], [1, 2]], 2, None),
            ([[0, 1], [0, 1]], 2, Some(0)),
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
            if let Some(patch) = hadamard {
                plan.instructions.push(Instruction::Hadamard { patch });
                expected.add_gate("h", vec![patch]);
            }
            for patches in pairs {
                plan.instructions.push(Instruction::Cz {
                    patches,
                    vertices: patches,
                });
                expected.add_gate("cz", patches.to_vec());
            }
            let (graph, inputs, outputs) = embed(&plan, 1000, 0).unwrap();
            if hadamard.is_some() {
                assert!(
                    graph
                        .blocks()
                        .all(|b| !matches!(b.kind(), BlockKind::PatchRotation(_)))
                );
            }
            let occupied_layers: std::collections::BTreeSet<_> = graph
                .pipes()
                .filter(|p| p.dir().is_spatial() && p.src().y == -1 && p.dst().y == -1)
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
        let (graph, inputs, outputs) = embed(&plan, 1000, 0).unwrap();
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
        let (graph, inputs, outputs) = embed(&plan, 100_000, 0).unwrap();
        let first_sources: Vec<_> = graph
            .blocks()
            .filter(|b| b.kind() == BlockKind::T)
            .map(Block::pos)
            .collect();
        let earliest = first_sources.iter().map(|p| p.z).min().unwrap();
        assert_eq!(first_sources.iter().filter(|p| p.z == earliest).count(), 2);
        assert!(
            embed(&plan, graph.block_count() - 1, 0)
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
            let plan = crate::spider::plan(&zx, None).unwrap();
            let (graph, inputs, outputs) =
                embed(&plan, 100_000, 0).unwrap_or_else(|e| panic!("{gates}: {e}"));
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
