//! The spatial-plane rotation is a single square of fixed-bulk checks.
//! Its opposite corners carry opposite temporal cap bases. Turning the split
//! between those caps by 90° from reset to readout transports both logical Paulis.

use bloq_circuit::PauliMap;
use bloq_graph::{Basis, Direction, PatchRotationKind, Pauli};
use glam::IVec2;

use super::grid::PatchData;
use super::observable::{
    INIT_CHUNK, MEAS_CHUNK, SelectedChunkMeasurements, SelectedGateway, include_l_shape_corner,
};
use super::utils::{Corner, checkerboard_basis, diagonal_cx_slots, tile_cx_schedule};
use crate::CompileError;
use crate::block::gateway::gateway_key_connectivity;
use crate::block::patch::{Patch, Tile};
use crate::block::{CompiledTemplate, LocalStabilizer};
use crate::signature::{Connectivity, LayerSchedule};

struct Layout {
    kind: PatchRotationKind,
    d: i32,
    lo: IVec2,
    hi: IVec2,
    end: IVec2,
    top_split_x: bool,
}

pub(super) fn readout_splits_x(kind: PatchRotationKind) -> bool {
    (kind.basis() == Basis::Z) == (kind.movement().x == kind.movement().y)
}

impl Layout {
    fn new(kind: PatchRotationKind, distance: u32) -> Result<Self, CompileError> {
        let step = kind.movement();
        let end = crate::compile::block_xy_offset(step, distance)?;
        let d = distance as i32;
        Ok(Self {
            kind,
            d,
            end,
            lo: end.min(IVec2::ZERO),
            hi: end.max(IVec2::ZERO) + IVec2::splat(2 * d),
            top_split_x: readout_splits_x(kind),
        })
    }

    fn center(&self, at_start: bool) -> IVec2 {
        IVec2::splat(self.d) + if at_start { IVec2::ZERO } else { self.end }
    }

    fn inside(&self, q: IVec2) -> bool {
        q.cmpgt(self.lo).all() && q.cmplt(self.hi).all()
    }

    fn at_start(&self, outward: Direction) -> bool {
        outward.to_ivec3().truncate().dot(self.kind.movement()) < 0
    }

    fn cap(&self, at_start: bool) -> Basis {
        if at_start {
            self.kind.basis()
        } else {
            self.kind.basis().flip()
        }
    }

    fn cap_at(&self, q: IVec2, split_x: bool) -> Basis {
        let mid = (self.lo + self.hi) / 2;
        let relative = (q - mid) * self.kind.movement();
        let across = if split_x { relative.x } else { relative.y };
        // The start cap owns the entire middle data line on both faces.
        self.cap(across <= 0)
    }

    /// Position along an arm's aperture, relative to its midpoint.
    fn along_arm(&self, q: IVec2, dir: Direction) -> i32 {
        let delta = q - self.center(self.at_start(dir));
        let normal = dir.to_ivec3().truncate();
        delta.perp_dot(normal).abs()
    }

    fn touches_data(&self, q: IVec2, conn: Connectivity) -> bool {
        self.inside(q)
            || conn.spatial_pipes().any(|dir| {
                let normal = dir.to_ivec3().truncate();
                let delta = q - self.center(self.at_start(dir));
                conn.has_open_edge(dir)
                    && delta.dot(normal) == self.d + 1
                    && delta.perp_dot(normal).abs() < self.d
            })
    }

    fn keep_tile(&self, m: IVec2, basis: Basis, conn: Connectivity) -> bool {
        [
            (m.x == self.lo.x, Direction::XMINUS),
            (m.x == self.hi.x, Direction::XPLUS),
            (m.y == self.lo.y, Direction::YMINUS),
            (m.y == self.hi.y, Direction::YPLUS),
        ]
        .into_iter()
        .filter(|(on, _)| *on)
        .all(|(_, dir)| {
            let along = self.along_arm(m, dir);
            if conn.has_hadamard(dir) && along <= self.d {
                return false; // The spatial wall owns the check on this segment.
            }
            (conn.has_open_edge(dir) && along < self.d) || basis != self.cap(self.at_start(dir))
        })
    }

    /// Turn the hook orientation to match the adjoining patch within the
    /// outward-facing triangle of each connected corner cell.
    fn horizontal_hook_basis(
        &self,
        m: IVec2,
        conn: Connectivity,
        flip_checkerboard: bool,
    ) -> Basis {
        for dir in conn.spatial_pipes() {
            let start = self.at_start(dir);
            let relative = m - self.center(start);
            let inward = if start {
                self.kind.movement()
            } else {
                -self.kind.movement()
            };
            if relative.abs().max_element() <= self.d && relative.dot(inward) <= 0 {
                let cap = self.cap(start);
                return if dir.to_ivec3().x != 0 {
                    cap.flip()
                } else {
                    cap
                };
            }
        }
        // Mirroring one axis about a distance-d patch center swaps the bulk checkerboard.
        if flip_checkerboard {
            Basis::Z
        } else {
            Basis::X
        }
    }

    fn patch(&self, conn: Connectivity, schedule: LayerSchedule) -> PatchData {
        let mut tiles = Vec::new();
        for y in (self.lo.y..=self.hi.y).step_by(2) {
            for x in (self.lo.x..=self.hi.x).step_by(2) {
                let m = IVec2::new(x, y);
                let basis = checkerboard_basis(m);
                if !self.keep_tile(m, basis, conn) {
                    continue;
                }
                let make_tile = |slots: &[Option<Corner>], prefix| {
                    let data = slots.iter().map(|slot| {
                        let q = m + (*slot)?.to_ivec2();
                        self.touches_data(q, conn).then_some(q)
                    });
                    Tile::new(basis, m, std::iter::repeat_n(None, prefix).chain(data))
                };
                let tile = if let LayerSchedule::PlanarHooks {
                    flip_x,
                    flip_y,
                    padded,
                } = schedule
                {
                    let horizontal = self.horizontal_hook_basis(m, conn, flip_x ^ flip_y);
                    make_tile(
                        &tile_cx_schedule(basis, horizontal)
                            .reflected_schedule(flip_x, flip_y, padded),
                        0,
                    )
                } else {
                    let first = schedule
                        .planar_start_slot()
                        .expect("planar schedule includes diagonal extraction");
                    make_tile(&diagonal_cx_slots(basis), first)
                };
                if tile.data_slots().iter().flatten().nth(1).is_some() {
                    tiles.push(tile);
                }
            }
        }
        let patch = Patch::from_sorted_tiles(tiles);
        let used = patch.data_set();
        let cap_map = |split_x| used.iter().map(|&q| (q, self.cap_at(q, split_x))).collect();
        (patch, cap_map(!self.top_split_x), cap_map(self.top_split_x))
    }

    fn gateway(&self, (patch, _, measured): &PatchData, conn: Connectivity) -> SelectedGateway {
        let mut gateway = SelectedGateway::default();
        for basis in [Basis::X, Basis::Z] {
            let start = basis == self.kind.basis();
            let center = self.center(start);
            let step = self.kind.movement() * if start { 1 } else { -1 };
            // Multiplying the checks in this quadrant turns the input string
            // into the output string. Their first readout supplies its sign.
            let mut records = vec![SelectedChunkMeasurements {
                chunk_index: INIT_CHUNK,
                qubits: patch
                    .tiles()
                    .iter()
                    .filter(|tile| {
                        tile.basis() == basis
                            && ((tile.measure_qubit() - center) * step)
                                .cmpgt(IVec2::ZERO)
                                .all()
                    })
                    .map(Tile::measure_qubit)
                    .collect(),
            }];
            if let Some(arm) = conn
                .spatial_pipes()
                .find(|&dir| self.at_start(dir) == start)
            {
                // The final logical string enters at the cap's aperture and follows
                // its same-basis readout stripe until it reaches the opposite wall.
                let along = if self.top_split_x {
                    IVec2::Y * step.y
                } else {
                    IVec2::X * step.x
                };
                let target = Direction::try_from(along.extend(0)).expect("cardinal stripe");
                let normal = arm.to_ivec3().truncate();
                let center_included =
                    arm == target.negate() || include_l_shape_corner(arm, target, basis);
                let on_ray =
                    |delta: IVec2, dir: IVec2| delta.perp_dot(dir) == 0 && delta.dot(dir) > 0;
                let mut qubits = measured
                    .keys()
                    .copied()
                    .filter(|&q| {
                        let delta = q - center;
                        (self.inside(q) || matches!(arm, Direction::XPLUS | Direction::YPLUS))
                            && ((delta == IVec2::ZERO && center_included)
                                || on_ray(delta, normal)
                                || on_ray(delta, along))
                    })
                    .collect::<Vec<_>>();
                qubits.sort_unstable_by_key(|q| (q.y, q.x));
                records.push(SelectedChunkMeasurements {
                    chunk_index: MEAS_CHUNK,
                    qubits,
                });
            }
            let key = LocalStabilizer::new(
                Pauli::from(basis),
                gateway_key_connectivity(conn.spatial_pipes()),
            );
            gateway.insert(key, records);
        }
        gateway
    }
}

pub(super) fn compile(
    kind: PatchRotationKind,
    conn: Connectivity,
    distance: u32,
    schedule: LayerSchedule,
) -> Result<CompiledTemplate, CompileError> {
    let layout = Layout::new(kind, distance)?;
    let data = layout.patch(conn, schedule);
    super::compile_patch_rounds(
        &data,
        schedule.is_extended().then_some(&data.0),
        distance,
        layout.gateway(&data, conn),
        |_| (PauliMap::empty(), PauliMap::empty()),
    )
}
