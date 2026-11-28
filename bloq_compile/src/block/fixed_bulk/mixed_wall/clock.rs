//! An exact, finite-domain schedule for overlapping cube and wall checks.
//! Cube/data precedence is inherited from the padded schedule. A table for
//! each wall star restricts its data slots to schedules admitting a cat circuit.

use glam::IVec2;
use std::collections::BTreeMap;

#[derive(Clone, Debug)]
pub(super) struct Cat {
    pub root: usize,
    pub final_wire: usize,
    pub gates: Vec<Option<(usize, usize)>>,
}

fn cat(slots: &[Vec<usize>], count: usize) -> Option<Cat> {
    let center = slots.len();
    let all = (1u16 << (center + 1)) - 1;
    let busy: Vec<u16> = (0..count)
        .map(|tick| {
            slots
                .iter()
                .enumerate()
                .filter(|(_, s)| s.contains(&tick))
                .fold(0, |m, (i, _)| m | (1 << i))
        })
        .collect();
    // A branch may drive data while it is in the cat state. It leaves
    // through one uncompute and cannot rejoin the cat later in the round.
    struct Search<'a> {
        center: usize,
        all: u16,
        busy: &'a [u16],
        end: usize,
    }
    impl Search<'_> {
        fn visit(
            &self,
            tick: usize,
            active: u16,
            seen: u16,
            path: &mut Vec<Option<(usize, usize)>>,
        ) -> bool {
            if tick == self.busy.len() {
                return seen == self.all && active == 1 << self.end;
            }
            if self.busy[tick] & !active != 0
                || (2 * (self.all ^ seen).count_ones() + active.count_ones() - 1) as usize
                    > self.busy.len() - tick
            {
                return false;
            }
            for leaf in 0..self.center {
                if self.busy[tick] & (1 << leaf) != 0 {
                    continue;
                }
                for (a, b) in [(self.center, leaf), (leaf, self.center)] {
                    if active & (1 << a) == 0 || (seen & !active) & (1 << b) != 0 {
                        continue;
                    }
                    path.push(Some((a, b)));
                    if self.visit(tick + 1, active ^ (1 << b), seen | (1 << b), path) {
                        return true;
                    }
                    path.pop();
                }
            }
            path.push(None);
            if self.visit(tick + 1, active, seen, path) {
                return true;
            }
            path.pop();
            false
        }
    }
    let mut ends: Vec<_> = (0..=center)
        .flat_map(|a| (0..=center).map(move |b| (a, b)))
        .collect();
    ends.sort_by_key(|&(a, b)| {
        (
            usize::from(a != center) + usize::from(b != center),
            a != center,
        )
    });
    for (root, final_wire) in ends {
        let mut gates = Vec::new();
        if (Search {
            center,
            all,
            busy: &busy,
            end: final_wire,
        })
        .visit(0, 1 << root, 1 << root, &mut gates)
        {
            return Some(Cat {
                root,
                final_wire,
                gates,
            });
        }
    }
    None
}

#[derive(Default)]
pub(super) struct Clock {
    by_qubit: crate::FxMap<IVec2, Vec<(u8, usize)>>,
    pub pairs: Vec<(IVec2, IVec2)>,
}

impl Clock {
    pub(super) fn add(&mut self, slot: u8, a: IVec2, b: IVec2) -> usize {
        let id = self.pairs.len();
        self.pairs.push((a, b));
        for q in [a, b] {
            self.by_qubit.entry(q).or_default().push((slot, id));
        }
        id
    }

    pub(super) fn schedule(
        &mut self,
        stars: &[Vec<Vec<usize>>],
        count: usize,
    ) -> Option<(Vec<usize>, Vec<Cat>)> {
        let mut order = Vec::new();
        for ops in self.by_qubit.values_mut() {
            ops.sort_unstable();
            for pair in ops.windows(2) {
                if pair[0].0 == pair[1].0 {
                    return None;
                }
                order.push((pair[0].1, pair[1].1));
            }
        }
        order.sort_unstable();
        let mut sets = BTreeMap::new();
        for star in stars {
            let widths = star.iter().map(Vec::len).collect::<Vec<_>>();
            sets.entry(widths.clone())
                .or_insert_with(|| options(&widths, count));
        }
        let tables: Vec<_> = stars
            .iter()
            .map(|star| {
                let widths = star.iter().map(Vec::len).collect::<Vec<_>>();
                (
                    star.iter().flatten().copied().collect::<Vec<_>>(),
                    &sets[&widths],
                )
            })
            .collect();
        // Bit t admits slot t. Prune conflicting slots using per-wire
        // precedence and each star's legal copy/uncompute schedules. Branch
        // on the smallest remaining star domain; the other events can then
        // take their earliest possible slots in precedence order.
        let full = (1u16 << count) - 1;
        let mut stack = vec![vec![full; self.pairs.len()]];
        while let Some(mut domains) = stack.pop() {
            let mut valid = true;
            loop {
                let mut changed = false;
                for &(a, b) in &order {
                    let low = domains[a].trailing_zeros();
                    let high = 15 - domains[b].leading_zeros();
                    let left = domains[a] & ((1 << high) - 1);
                    let right = domains[b] & (!1u16 << low);
                    if left == 0 || right == 0 {
                        valid = false;
                        break;
                    }
                    changed |= left != domains[a] || right != domains[b];
                    domains[a] = left;
                    domains[b] = right;
                }
                if !valid {
                    break;
                }
                for (vars, table) in &tables {
                    let mut support = vec![0u16; vars.len()];
                    for values in table.keys() {
                        if vars
                            .iter()
                            .zip(values)
                            .all(|(&v, &t)| domains[v] & (1 << t) != 0)
                        {
                            for (s, &t) in support.iter_mut().zip(values) {
                                *s |= 1 << t;
                            }
                        }
                    }
                    if support.contains(&0) {
                        valid = false;
                        break;
                    }
                    for (&v, s) in vars.iter().zip(support) {
                        changed |= s != domains[v];
                        domains[v] = s;
                    }
                }
                if !valid || !changed {
                    break;
                }
            }
            if !valid {
                continue;
            }
            let next = tables
                .iter()
                .flat_map(|(vars, _)| vars)
                .copied()
                .filter(|&v| domains[v].count_ones() > 1)
                .min_by_key(|&v| domains[v].count_ones());
            if let Some(v) = next {
                for t in (0..count).rev().filter(|t| domains[v] & (1 << t) != 0) {
                    let mut branch = domains.clone();
                    branch[v] = 1 << t;
                    stack.push(branch);
                }
            } else {
                let times = domains
                    .iter()
                    .map(|m| m.trailing_zeros() as usize)
                    .collect::<Vec<_>>();
                let cats = tables
                    .iter()
                    .map(|(vars, table)| {
                        table[&vars.iter().map(|&v| times[v]).collect::<Vec<_>>()].clone()
                    })
                    .collect();
                return Some((times, cats));
            }
        }
        None
    }
}

fn options(widths: &[usize], count: usize) -> BTreeMap<Vec<usize>, Cat> {
    fn enumerate(
        widths: &[usize],
        count: usize,
        rows: &mut Vec<Vec<usize>>,
        result: &mut BTreeMap<Vec<usize>, Cat>,
    ) {
        if rows.len() == widths.len() {
            if let Some(cat) = cat(rows, count) {
                result.insert(rows.iter().flatten().copied().collect(), cat);
            }
            return;
        }
        for mask in 0..(1u16 << count) {
            if mask.count_ones() as usize != widths[rows.len()] {
                continue;
            }
            rows.push((0..count).filter(|t| mask & (1 << t) != 0).collect());
            enumerate(widths, count, rows, result);
            rows.pop();
        }
    }
    let mut result = BTreeMap::new();
    enumerate(widths, count, &mut Vec::new(), &mut result);
    result
}
