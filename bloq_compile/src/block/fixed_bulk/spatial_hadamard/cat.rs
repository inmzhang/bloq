//! Grow and optionally uncompute a nearest-neighbor cat on the link and its branches.
//! The surviving wires are measured in X and their product is the check syndrome.

#[derive(Clone, Debug)]
pub(super) struct Cat {
    pub root: usize,
    pub final_mask: u16,
    pub gates: Vec<Option<(usize, usize)>>,
}

pub(super) fn cat(slots: &[Vec<usize>], count: usize) -> Option<Cat> {
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
        end: u16,
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
                return seen == self.all && active == self.end;
            }
            if self.busy[tick] & !active != 0
                || ((self.all ^ seen).count_ones()
                    + ((active | (self.all ^ seen)) & !self.end).count_ones())
                    as usize
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
    let mut ends: Vec<_> = (1..=all)
        .flat_map(|mask| (0..=center).map(move |root| (root, mask)))
        .collect();
    ends.sort_by_key(|&(root, mask)| {
        (
            mask.count_ones(),
            (0..center)
                .filter(|&i| mask & (1 << i) != 0 && slots[i].len() == 2)
                .count(),
            root != center,
            mask & (1 << center) == 0,
            mask,
            root,
        )
    });
    for (root, final_mask) in ends {
        let mut gates = Vec::new();
        if (Search {
            center,
            all,
            busy: &busy,
            end: final_mask,
        })
        .visit(0, 1 << root, 1 << root, &mut gates)
        {
            return Some(Cat {
                root,
                final_mask,
                gates,
            });
        }
    }
    None
}
