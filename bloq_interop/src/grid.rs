//! Fixed patch islands and bounded, deterministic planar routing.

use std::cell::RefCell;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

use glam::IVec2;
use rustc_hash::{FxHashMap, FxHashSet};

/// Two-dimensional data-patch placement and routing bounds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GridOptions {
    /// Maximum patch locations per row, filled in row-major order.
    /// Zero (the default) resolves to a near-square grid for the allocated sites.
    pub columns: usize,
    /// Pitch between patch homes on both axes, in graph cells (at least 4).
    /// Each home reserves room for both patch frames and local T/Y resources.
    pub spacing: u32,
    /// Extra routing canvas around the patch islands, in graph cells (positive).
    /// Only cells actually used by geometry contribute to the final footprint.
    pub routing_margin: u32,
    /// Maximum expanded A* states per route search (positive).
    pub max_route_expansions: usize,
}

impl Default for GridOptions {
    fn default() -> Self {
        Self {
            columns: 0,
            spacing: 4,
            routing_margin: 2,
            max_route_expansions: 100_000,
        }
    }
}

impl GridOptions {
    pub(crate) fn validate(self) -> Result<(), String> {
        if self.spacing < 4 || self.routing_margin == 0 || self.max_route_expansions == 0 {
            return Err(
                "grid requires spacing >= 4, routing_margin > 0 and max_route_expansions > 0"
                    .into(),
            );
        }
        Ok(())
    }
}

pub(crate) struct Grid {
    homes: Vec<IVec2>,
    islands: FxHashSet<IVec2>,
    min: IVec2,
    max: IVec2,
    options: GridOptions,
    /// Route-search maps, cleared but not reallocated between routes.
    search: RefCell<(FxHashMap<IVec2, i64>, FxHashMap<IVec2, IVec2>)>,
}

impl Grid {
    pub(crate) fn new(mut options: GridOptions, slots: usize) -> Result<Self, String> {
        options.validate()?;
        if options.columns == 0 {
            let root = slots.isqrt();
            options.columns = (root + usize::from(root * root < slots)).max(1);
        }
        let pitch = usize::try_from(options.spacing).map_err(|e| e.to_string())?;
        let margin = i32::try_from(options.routing_margin)
            .map_err(|_| "routing margin exceeds spatial coordinate range")?;
        let coordinate = |index: usize| {
            index
                .checked_mul(pitch)
                .and_then(|x| i32::try_from(x).ok())
                .ok_or_else(|| "grid exceeds spatial coordinate range".to_string())
        };
        let mut homes = Vec::with_capacity(slots);
        let mut islands = FxHashSet::default();
        let mut max = IVec2::ZERO;
        for slot in 0..slots {
            let home = IVec2::new(
                coordinate(slot % options.columns)?,
                coordinate(slot / options.columns)?,
            );
            let corner = IVec2::new(
                home.x.checked_add(1).ok_or("grid x overflow")?,
                home.y.checked_add(2).ok_or("grid y overflow")?,
            );
            max = max.max(corner);
            homes.push(home);
            // Include vacant/future lifetimes and direct wires, not only live patches.
            for dx in 0..=1 {
                for dy in 0..=2 {
                    islands.insert(home + IVec2::new(dx, dy));
                }
            }
        }
        max = IVec2::new(
            max.x
                .checked_add(margin)
                .ok_or("routing canvas x overflow")?,
            max.y
                .checked_add(margin)
                .ok_or("routing canvas y overflow")?,
        );
        Ok(Self {
            homes,
            islands,
            min: IVec2::splat(-margin),
            max,
            options,
            search: RefCell::default(),
        })
    }

    pub(crate) fn options(&self) -> GridOptions {
        self.options
    }

    pub(crate) fn bounds(&self) -> [IVec2; 2] {
        [self.min, self.max]
    }

    pub(crate) fn home(&self, slot: usize) -> Result<IVec2, String> {
        self.homes
            .get(slot)
            .copied()
            .ok_or_else(|| "unknown grid slot".into())
    }

    /// Shortest orthogonal route; temporary occupancy can defer an otherwise legal path.
    pub(crate) fn path(
        &self,
        from: IVec2,
        to: IVec2,
        busy: impl Fn(IVec2) -> bool,
    ) -> Result<Option<Vec<IVec2>>, String> {
        let distance = |p: IVec2| i64::from(p.x.abs_diff(to.x)) + i64::from(p.y.abs_diff(to.y));
        let mut frontier = BinaryHeap::from([Reverse((
            distance(from),
            distance(from),
            0_i64,
            from.x,
            from.y,
        ))]);
        let mut search = self.search.borrow_mut();
        let (costs, parents) = &mut *search;
        costs.clear();
        parents.clear();
        costs.insert(from, 0_i64);
        let mut expanded = 0;
        while let Some(Reverse((_, _, cost, x, y))) = frontier.pop() {
            let current = IVec2::new(x, y);
            if costs.get(&current) != Some(&cost) {
                continue;
            }
            if expanded == self.options.max_route_expansions {
                return Err(format!(
                    "routing search exhausted {} expanded states",
                    self.options.max_route_expansions
                ));
            }
            expanded += 1;
            if current == to {
                let mut current = to;
                let mut path = vec![current];
                while current != from {
                    current = parents[&current];
                    path.push(current);
                }
                path.reverse();
                return Ok(Some(path));
            }
            for step in [IVec2::X, IVec2::Y, IVec2::NEG_X, IVec2::NEG_Y] {
                let (Some(x), Some(y)) = (x.checked_add(step.x), y.checked_add(step.y)) else {
                    continue;
                };
                let next = IVec2::new(x, y);
                if next.cmplt(self.min).any()
                    || next.cmpgt(self.max).any()
                    || self.islands.contains(&next)
                    || busy(next)
                    || costs.get(&next).is_some_and(|&old| old <= cost + 1)
                {
                    continue;
                }
                costs.insert(next, cost + 1);
                parents.insert(next, current);
                let remaining = distance(next);
                frontier.push(Reverse((cost + 1 + remaining, remaining, cost + 1, x, y)));
            }
        }
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn automatic_width_and_spatial_coordinate_bounds() {
        let grid = Grid::new(GridOptions::default(), 3).unwrap();
        assert_eq!(grid.options().columns, 2);
        assert_eq!(
            (0..3)
                .map(|slot| grid.home(slot).unwrap())
                .collect::<Vec<_>>(),
            [IVec2::ZERO, IVec2::new(4, 0), IVec2::new(0, 4)]
        );
        // A maximum row width does not allocate or reserve empty columns.
        Grid::new(
            GridOptions {
                columns: usize::MAX,
                ..Default::default()
            },
            3,
        )
        .unwrap();
        for options in [
            GridOptions {
                spacing: u32::MAX,
                ..Default::default()
            },
            GridOptions {
                routing_margin: u32::MAX,
                ..Default::default()
            },
        ] {
            assert!(Grid::new(options, 3).is_err());
        }
    }

    #[test]
    fn routes_detour_around_reserved_and_busy_cells_deterministically() {
        let grid = Grid::new(
            GridOptions {
                columns: 2,
                routing_margin: 2,
                ..Default::default()
            },
            4,
        )
        .unwrap();
        let (from, to) = (IVec2::new(1, -1), IVec2::new(4, 3));
        let busy = IVec2::new(2, -1);
        let direct = grid.path(from, to, |_| false).unwrap().unwrap();
        assert!(direct.contains(&busy));
        let detour = grid.path(from, to, |p| p == busy).unwrap().unwrap();
        assert_eq!(detour, grid.path(from, to, |p| p == busy).unwrap().unwrap());
        assert!(!detour.contains(&busy));
        assert!(detour.iter().all(|p| !grid.islands.contains(p)));
        assert!(
            detour
                .windows(2)
                .all(|p| (p[1] - p[0]).abs().element_sum() == 1)
        );
        assert!(grid.path(from, to, |_| true).unwrap().is_none());
    }
}
