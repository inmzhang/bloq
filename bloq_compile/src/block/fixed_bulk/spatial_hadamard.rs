//! Plan the cube and pipe circuits of one spatial component containing Hadamard walls together.
//! Members retain separate translated templates and a common entangling clock.

mod cat;
mod geometry;
pub(crate) use geometry::{SpatialHadamardKey, WallSide};
mod schedule;

use bloq_circuit::{
    BodyId, Chunk, ChunkOrLoop, CoordCircuit, Flow, GateType, Op, PauliBasis, PauliMap,
};
use bloq_graph::{Basis, CubeKind, Pauli, UDirection};
use bloq_ir::lowering::ReadoutFrame;
use glam::{IVec2, IVec3};
use std::sync::Arc;

use super::utils::{reset_gate, sorted_batches};
use super::{build_cube_patch, compile_scheduled_cube, cube_bulk_repetitions};
use crate::block::gateway::{ChunkMeasurements, GatewayEntry};
use crate::block::measurements::MeasurementIndex;
use crate::block::patch::{Patch, Tile};
use crate::block::{CompiledTemplate, LoweringTemplate, ObservableGateway};
use crate::lower::SpatialPipeRef;
use crate::signature::{Connectivity, LayerSchedule};
use crate::{CompileError, FxMap};
use cat::Cat;
use geometry::{Face, collapse_map, crossing_key, wall_tiles};
use schedule::Clock;

const SLOTS: usize = 5;

pub(crate) struct Cube {
    pub position: IVec3,
    pub origin: IVec2,
    pub kind: CubeKind,
    pub connectivity: Connectivity,
    pub rounds: u32,
}
pub(crate) struct Pipe {
    pub id: SpatialPipeRef,
    pub origin: IVec2,
    pub key: SpatialHadamardKey,
}

#[derive(Clone)]
struct Branch {
    anc: IVec2,
    basis: Basis,
    data: Vec<IVec2>,
    owner: usize,
    events: Vec<usize>,
}

struct Star {
    arms: Vec<Branch>,
    // One entry per incident wall, in that wall's crossing basis.
    pipes: Vec<(usize, Basis)>,
    link: IVec2,
}

impl Star {
    fn boundary(&self, collapse: &FxMap<IVec2, Basis>) -> Option<(PauliMap, Vec<IVec2>)> {
        let mut remaining = PauliMap::empty();
        let mut records = Vec::new();
        for arm in &self.arms {
            for &q in &arm.data {
                if let Some(&basis) = collapse.get(&q) {
                    if basis != arm.basis {
                        return None;
                    }
                    records.push(q);
                } else {
                    remaining.insert(q, Pauli::from(arm.basis));
                }
            }
        }
        Some((remaining, records))
    }
}

pub(crate) struct Templates {
    pub cubes: Vec<(IVec3, CompiledTemplate)>,
    pub pipes: Vec<(SpatialPipeRef, CompiledTemplate)>,
}

pub(crate) fn compile(
    distance: u32,
    cubes: &[Cube],
    pipes: &[Pipe],
) -> Result<Templates, CompileError> {
    let mut clock = Clock::default();
    let mut patches = Vec::new();
    for cube in cubes {
        let (patch, init, meas) = build_cube_patch(
            cube.kind,
            cube.connectivity,
            distance,
            LayerSchedule::Padded,
        );
        let events: Vec<Vec<usize>> = patch
            .tiles()
            .iter()
            .map(|tile| {
                tile.data_slots()
                    .iter()
                    .flatten()
                    .map(|&q| clock.add(q + cube.origin))
                    .collect()
            })
            .collect();
        patches.push((patch, init, meas, events));
    }
    let geometry = wall_geometry(distance, pipes, &mut clock)?;
    let stars = &geometry.stars;
    let checks = patches.iter().flat_map(|(patch, _, _, events)| {
        patch
            .tiles()
            .iter()
            .zip(events)
            .map(|(tile, ids)| (tile.basis(), ids.as_slice()))
    });
    let (times, cats) =
        clock
            .solve(stars, pipes, checks)
            .ok_or(CompileError::SpatialWallSchedule {
                pos: cubes[0].position,
            })?;
    let mut compiled = Templates {
        cubes: Vec::new(),
        pipes: Vec::new(),
    };
    for (cube, (patch, init, meas, events)) in cubes.iter().zip(patches) {
        let tiles = patch
            .tiles()
            .iter()
            .zip(events)
            .map(|(tile, events)| {
                let mut data = vec![None; SLOTS];
                for id in events {
                    data[times[id]] = Some(clock.data[id] - cube.origin);
                }
                Tile::new(tile.basis(), tile.measure_qubit(), data)
            })
            .collect();
        let template = compile_scheduled_cube(
            cube.kind,
            cube.connectivity,
            distance,
            cube.rounds,
            (Patch::from_sorted_tiles(tiles), init, meas),
        )?;
        compiled.cubes.push((cube.position, template));
    }
    bind_shared_corners(&geometry, &cats, cubes, pipes, &mut compiled)?;
    for (p, pipe) in pipes.iter().enumerate() {
        let round = PipeRound {
            p,
            pipe,
            stars,
            cats: &cats,
            times: &times,
            joint_checks: geometry.joint_checks[p],
        };
        compiled
            .pipes
            .push((pipe.id, round.compile(&geometry.init, &geometry.meas)?));
    }
    Ok(compiled)
}

struct WallGeometry {
    stars: Vec<Star>,
    init: FxMap<IVec2, Basis>,
    meas: FxMap<IVec2, Basis>,
    joint_checks: Vec<bool>,
}

/// Merge tile halves that share a corner ancilla and establish a compatible
/// interaction order at every cube and wall data qubit.
fn wall_geometry(
    distance: u32,
    pipes: &[Pipe],
    clock: &mut Clock,
) -> Result<WallGeometry, CompileError> {
    let mut joint_checks = Vec::with_capacity(pipes.len());
    let mut groups = petgraph::unionfind::UnionFind::<usize>::new(0);
    let mut anchors = FxMap::default();
    let mut rows = Vec::new();
    let mut init = FxMap::default();
    let mut meas = FxMap::default();
    for (owner, pipe) in pipes.iter().enumerate() {
        let tiles = wall_tiles(distance, pipe.key);
        let local_init = collapse_map(distance, pipe.key, &tiles, Face::Init);
        joint_checks.push(pipe.key.introduces_joint_checks(&local_init));
        init.extend(local_init.into_iter().map(|(q, b)| (q + pipe.origin, b)));
        meas.extend(
            collapse_map(distance, pipe.key, &tiles, Face::Meas)
                .into_iter()
                .map(|(q, b)| (q + pipe.origin, b)),
        );
        for tile in tiles {
            let i = rows.len();
            groups.new_set();
            for anc in [tile.minus, tile.plus] {
                if let Some(j) = anchors.insert(anc + pipe.origin, i) {
                    groups.union(i, j);
                }
            }
            rows.push((owner, tile));
        }
    }
    let mut grouped = std::collections::BTreeMap::<usize, Vec<_>>::new();
    for (i, row) in rows.into_iter().enumerate() {
        grouped.entry(groups.find(i)).or_default().push(row);
    }
    let mut stars = Vec::new();
    for mut group in grouped.into_values() {
        group.sort_by_key(|(p, _)| (pipes[*p].key.axis != UDirection::X, *p));
        let mut star = Star {
            arms: Vec::new(),
            pipes: Vec::new(),
            link: IVec2::ZERO,
        };
        for (owner, tile) in group {
            star.pipes.push((owner, tile.basis));
            let offset = pipes[owner].origin;
            for (anc, basis, qubits) in [
                (tile.minus, tile.basis, tile.minus_data),
                (tile.plus, tile.basis.flip(), tile.plus_data),
            ] {
                let anc = anc + offset;
                if let Some(common) = star.arms.iter().find(|a| a.anc == anc) {
                    if common.basis != basis
                        || common.data.len() != qubits.len()
                        || !qubits.iter().all(|q| common.data.contains(&(*q + offset)))
                    {
                        return Err(CompileError::SpatialWallSchedule {
                            pos: pipes[owner].id.src,
                        });
                    }
                    continue;
                }
                star.arms.push(Branch {
                    anc,
                    basis,
                    data: qubits.iter().map(|q| *q + offset).collect(),
                    owner,
                    events: Vec::new(),
                });
            }
        }
        let min = star
            .arms
            .iter()
            .map(|a| a.anc)
            .reduce(IVec2::min)
            .expect("wall has endpoints");
        star.link = min + IVec2::ONE;
        for arm in &mut star.arms {
            // Both halves index data in the same direction along the seam.
            arm.data.sort_by_key(|q| (std::cmp::Reverse(q.y), q.x));
            arm.events = arm.data.iter().map(|&q| clock.add(q)).collect();
        }
        stars.push(star);
    }
    Ok(WallGeometry {
        stars,
        init,
        meas,
        joint_checks,
    })
}

/// Attach the physical corner record to the hub's first-round observable gateway.
fn bind_shared_corners(
    geometry: &WallGeometry,
    cats: &[Cat],
    cubes: &[Cube],
    pipes: &[Pipe],
    compiled: &mut Templates,
) -> Result<(), CompileError> {
    for (star, cat) in geometry
        .stars
        .iter()
        .zip(cats)
        .filter(|(s, _)| s.pipes.len() > 1)
    {
        // The hub sees both arms even when the compiler binds one source
        // position at a time (guarded readouts, module linking and proxies).
        let first = pipes[star.pipes[0].0].id;
        let hub = [first.src, first.dst]
            .into_iter()
            .find(|&pos| {
                star.pipes.iter().all(|&(p, _)| {
                    let id = pipes[p].id;
                    id.src == pos || id.dst == pos
                })
            })
            .ok_or(CompileError::SpatialWallSchedule { pos: first.src })?;
        let mut arms = Vec::new();
        for &(p, basis) in &star.pipes {
            let pipe = &pipes[p];
            let joint = geometry.joint_checks[p];
            if star.boundary(&geometry.init).is_some()
                && !(joint && basis == pipe.key.boundary_basis)
            {
                continue;
            }
            let (minus, plus) = pipe.id.arm_directions();
            arms.push(if hub == pipe.id.src {
                (plus, Pauli::from(basis))
            } else {
                (minus, Pauli::from(basis.flip()))
            });
        }
        if arms.is_empty() {
            continue;
        }
        let index = cubes
            .iter()
            .position(|c| c.position == hub)
            .expect("wall hub cube");
        let t = Arc::make_mut(&mut compiled.cubes[index].1);
        let c = &mut Arc::make_mut(&mut t.program_template).circuit;
        for wire in (0..=star.arms.len()).filter(|&i| cat.final_mask & (1 << i) != 0) {
            let q = star.arms.get(wire).map_or(star.link, |a| a.anc) - cubes[index].origin;
            let id = c.reserve_measurement_id(q);
            let ops = c.body_mut(c.entry_body()).expect("cube entry").ops_mut();
            let index = ops
                .iter()
                .position(|op| matches!(op, Op::Measure { .. }))
                .expect("cube first readout");
            ops.insert(
                index,
                Op::Measure {
                    basis: PauliBasis::X,
                    qubits: vec![q],
                    measurements: vec![id],
                    flip_probability: 0.0,
                },
            );
            t.observable_gateway.bind_wall_corner(&arms, id);
        }
    }
    Ok(())
}

struct PipeRound<'a> {
    p: usize,
    pipe: &'a Pipe,
    stars: &'a [Star],
    cats: &'a [Cat],
    times: &'a [usize],
    joint_checks: bool,
}

/// Flag wire to the last data interaction on its branch.
type BranchFrames = FxMap<IVec2, (IVec2, IVec2, PauliBasis)>;

impl PipeRound<'_> {
    fn compile(
        &self,
        init: &FxMap<IVec2, Basis>,
        meas: &FxMap<IVec2, Basis>,
    ) -> Result<CompiledTemplate, CompileError> {
        let empty = FxMap::default();
        let (first, index, _) = self.chunk(init, &empty)?;
        let key = self.pipe.key;
        let mut gateway = ObservableGateway::new();
        for basis in [Basis::X, Basis::Z] {
            let mut entry = GatewayEntry::default();
            let mut records = Vec::new();
            for (star, cat) in self.stars.iter().zip(self.cats) {
                if star.pipes.len() == 1
                    && star.pipes.contains(&(self.p, basis))
                    && ((self.joint_checks && basis == key.boundary_basis)
                        || star.boundary(init).is_none())
                {
                    records.extend(
                        (0..=star.arms.len())
                            .filter(|&i| cat.final_mask & (1 << i) != 0)
                            .map(|i| index.expect_measurement(self.wire(star, i))),
                    );
                }
            }
            if !records.is_empty() {
                entry.measurements.push(ChunkMeasurements {
                    chunk_index: 0,
                    measurements: records,
                });
            }
            gateway.insert(crossing_key(key.axis, Pauli::from(basis)), entry);
        }
        let (bulk, _, frames) = self.chunk(&empty, &empty)?;
        let (last, _, _) = self.chunk(&empty, meas)?;
        let mut chunks = vec![ChunkOrLoop::Single(Box::new(first))];
        let repetitions = cube_bulk_repetitions(key.rounds);
        if repetitions > 0 {
            chunks.push(ChunkOrLoop::Loop {
                body: vec![bulk],
                repetitions,
            });
        }
        chunks.push(ChunkOrLoop::Single(Box::new(last)));
        let mut result = LoweringTemplate::from_chunks(chunks, gateway)?;
        let template = Arc::make_mut(&mut result.program_template);
        for i in 0..template.circuit.body_count() {
            let body = template
                .circuit
                .body(BodyId(i as u32))
                .expect("allocated body");
            for op in body.ops() {
                if let Op::Measure {
                    qubits,
                    measurements,
                    basis: PauliBasis::Z,
                    ..
                } = op
                {
                    for (q, &flag) in qubits.iter().zip(measurements) {
                        if let Some(&(ancilla, target, pauli)) = frames.get(q) {
                            template.readout_frames.push(ReadoutFrame {
                                flag,
                                ancilla,
                                target,
                                pauli,
                            });
                        }
                    }
                }
            }
        }
        Ok(Arc::new(result))
    }

    fn wire(&self, s: &Star, index: usize) -> IVec2 {
        s.arms.get(index).map_or(s.link, |a| a.anc) - self.pipe.origin
    }

    fn chunk(
        &self,
        init: &FxMap<IVec2, Basis>,
        meas: &FxMap<IVec2, Basis>,
    ) -> Result<(Chunk, MeasurementIndex, BranchFrames), CompileError> {
        let mut c = CoordCircuit::new();
        let mut reads = MeasurementIndex::default();
        let mut ops = vec![Vec::new(); SLOTS];
        let mut anc_measure = Vec::new();
        let mut frames = FxMap::default();
        let mut init_data = FxMap::default();
        let mut meas_data = FxMap::default();
        let mut owned = Vec::new();
        for (star, cat) in self.stars.iter().zip(self.cats) {
            if !star.pipes.iter().any(|(p, _)| *p == self.p) {
                continue;
            }
            let primary = star.pipes[0].0;
            let syndrome: Vec<_> = (0..=star.arms.len())
                .filter(|&i| cat.final_mask & (1 << i) != 0)
                .map(|i| self.wire(star, i))
                .collect();
            if primary == self.p {
                owned.push((star, syndrome));
            }
            for i in 0..=star.arms.len() {
                let q = self.wire(star, i);
                let owner = star.arms.get(i).map_or(primary, |a| a.owner);
                if owner == self.p {
                    c.do_gate(
                        if i == cat.root {
                            GateType::RX
                        } else {
                            GateType::RZ
                        },
                        [q],
                    )?;
                }
                // Participating instances alias the same physical readout.
                // The hub owns the shared corner gateway.
                anc_measure.push((
                    q,
                    if cat.final_mask & (1 << i) != 0 {
                        PauliBasis::X
                    } else {
                        PauliBasis::Z
                    },
                ));
            }
            let mut last_flag = vec![None; star.arms.len()];
            let mut active = 1u16 << cat.root;
            for (slot, gate) in cat.gates.iter().enumerate() {
                if let Some((a, b)) = *gate {
                    let arm = a.min(b);
                    if active & (1 << b) != 0 {
                        last_flag[arm] = Some(self.wire(star, b));
                    }
                    active ^= 1 << b;
                    if star.arms[arm].owner == self.p {
                        ops[slot].push((GateType::CX, self.wire(star, a), self.wire(star, b)));
                    }
                }
            }
            for (i, arm) in star.arms.iter().enumerate() {
                for (&q, &event) in arm.data.iter().zip(&arm.events) {
                    let local = q - self.pipe.origin;
                    if primary == self.p || arm.owner == self.p {
                        if let Some(b) = init.get(&q) {
                            init_data.insert(local, *b);
                        }
                        if let Some(b) = meas.get(&q) {
                            meas_data.insert(local, *b);
                        }
                    }
                    if arm.owner == self.p {
                        ops[self.times[event]].push((
                            if arm.basis == Basis::X {
                                GateType::CX
                            } else {
                                GateType::CZ
                            },
                            arm.anc - self.pipe.origin,
                            local,
                        ));
                    }
                }
                if arm.owner == self.p
                    && arm.data.len() == 2
                    && let Some(flag) = last_flag[i]
                {
                    let last_data = if self.times[arm.events[1]] > self.times[arm.events[0]] {
                        arm.data[1]
                    } else {
                        arm.data[0]
                    };
                    frames.insert(
                        flag,
                        (
                            arm.anc - self.pipe.origin,
                            last_data - self.pipe.origin,
                            PauliBasis::from(arm.basis),
                        ),
                    );
                }
            }
        }
        for (basis, qs) in sorted_batches(Some(&init_data), [Basis::X, Basis::Z]) {
            c.do_gate(reset_gate(basis), qs)?;
        }
        c.tick();
        for moment in ops {
            for (gate, a, b) in moment {
                c.do_gate(gate, [a, b])?;
            }
            c.tick();
        }
        for (q, b) in anc_measure {
            let id = c.measure(b, [q])[0];
            reads.record(q, id);
        }
        for (b, qs) in sorted_batches(Some(&meas_data), [Basis::X, Basis::Z]) {
            for (&q, id) in qs
                .iter()
                .zip(c.measure(PauliBasis::from(b), qs.iter().copied()))
            {
                reads.record(q, id);
            }
        }
        let mut flows = Vec::new();
        for (star, syndrome) in owned {
            let ids: Vec<_> = syndrome
                .into_iter()
                .map(|q| reads.expect_measurement(q))
                .collect();
            let center = star.link - self.pipe.origin;
            if let Some((start, _)) = star.boundary(init) {
                flows.push(
                    Flow::new(start.translated(-self.pipe.origin), PauliMap::empty())
                        .with_measurements(ids.iter().copied())
                        .with_center(center),
                );
            }
            if let Some((end, collapsed)) = star.boundary(meas) {
                flows.push(
                    Flow::new(PauliMap::empty(), end.translated(-self.pipe.origin))
                        .with_measurements(
                            collapsed
                                .into_iter()
                                .map(|q| reads.expect_measurement(q - self.pipe.origin))
                                .chain(ids),
                        )
                        .with_center(center),
                );
            }
        }
        Ok((Chunk { circuit: c, flows }, reads, frames))
    }
}
