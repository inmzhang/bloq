//! Five-slot scheduling of ordinary plaquettes and multi-branch wall checks.
use super::{Basis, Cat, FxMap, IVec2, Pipe, SLOTS, Star, UDirection, cat};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::{Arc, Mutex, OnceLock};

#[derive(Default)]
pub(super) struct Clock {
    pub(super) data: Vec<IVec2>,
}
impl Clock {
    pub(super) fn add(&mut self, data: IVec2) -> usize {
        let id = self.data.len();
        self.data.push(data);
        id
    }
}

#[derive(Clone)]
struct Candidate {
    slots: Vec<u8>,
    cat: Option<Cat>,
}
type Choices = Arc<Vec<Candidate>>;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum Shape {
    Cube {
        width: usize,
        flexible: bool,
    },
    Wall {
        widths: Vec<usize>,
        protected: bool,
        axis: UDirection,
    },
}

fn choices(shape: Shape) -> Choices {
    static CACHE: OnceLock<Mutex<FxMap<Shape, Choices>>> = OnceLock::new();
    let cache = CACHE.get_or_init(Mutex::default);
    if let Some(value) = cache.lock().expect("schedule cache").get(&shape) {
        return Arc::clone(value);
    }
    let widths = match &shape {
        Shape::Cube { width, .. } => vec![*width],
        Shape::Wall { widths, .. } => widths.clone(),
    };
    let mut branch = Vec::new();
    for (i, &w) in widths.iter().enumerate() {
        branch.extend(std::iter::repeat_n(i, w));
    }
    let mut result = Vec::new();
    // Each check sees at most four ordinary data or three wall branches.
    let mut slots = vec![0u8; branch.len()];
    let mut cats = FxMap::<Vec<u8>, Option<Cat>>::default();
    for index in 0..SLOTS.pow(slots.len() as u32) {
        let mut value = index;
        let mut masks = vec![0u8; widths.len()];
        let mut collision = false;
        for (j, t) in slots.iter_mut().enumerate() {
            *t = (value % SLOTS) as u8;
            value /= SLOTS;
            let m = &mut masks[branch[j]];
            if *m & (1 << *t) != 0 {
                collision = true;
                break;
            }
            *m |= 1 << *t;
        }
        if collision {
            continue;
        }
        let cat = match &shape {
            Shape::Cube { width, flexible } => {
                let ordered = slots.windows(2).all(|w| w[0] < w[1]);
                if !ordered
                    && (!flexible
                        || (*width == 4
                            && !(slots[0].max(slots[1]) < slots[2].min(slots[3])
                                || slots[0].min(slots[1]) > slots[2].max(slots[3]))))
                {
                    continue;
                }
                None
            }
            Shape::Wall { protected, .. } => {
                // Align the flagged corrections across the seam: the neighboring
                // wall syndrome shared by the final data pair then cancels.
                if *protected && (slots[0] < slots[1]) != (slots[2] < slots[3]) {
                    continue;
                }

                let c = cats.entry(masks.clone()).or_insert_with(|| {
                    let data = masks
                        .iter()
                        .map(|m| (0..SLOTS).filter(|t| m & (1 << t) != 0).collect())
                        .collect::<Vec<_>>();
                    cat::cat(&data, SLOTS)
                });
                let Some(cat) = c.as_ref() else { continue };
                if *protected && cat.final_mask.count_ones() != 1 {
                    continue;
                }
                Some(cat.clone())
            }
        };
        result.push(Candidate {
            slots: slots.clone(),
            cat,
        });
    }
    result.sort_by_cached_key(|v| {
        let (violations, target) = match &shape {
            Shape::Cube { .. } => (
                v.slots.windows(2).filter(|w| w[0] > w[1]).count(),
                (0..v.slots.len() as u8).collect::<Vec<_>>(),
            ),
            Shape::Wall {
                widths,
                protected,
                axis,
            } => {
                let prefer = if *protected {
                    if *axis == UDirection::Y {
                        [[3, 2], [1, 2]]
                    } else {
                        [[1, 2], [3, 2]]
                    }
                } else {
                    [[3, 4], [3, 4]]
                };
                let target = widths
                    .iter()
                    .enumerate()
                    .flat_map(|(i, &len)| prefer[i.min(1)].into_iter().take(len))
                    .collect();
                (0, target)
            }
        };
        (
            violations,
            v.slots
                .iter()
                .zip(target)
                .map(|(a, b)| a.abs_diff(b) as usize)
                .sum::<usize>(),
        )
    });
    let result = Arc::new(result);
    cache
        .lock()
        .expect("schedule cache")
        .insert(shape, Arc::clone(&result));
    result
}

struct Check {
    events: Vec<(usize, Basis)>,
    choices: Choices,
}
#[derive(Clone, Copy)]
struct Intersection {
    left: usize,
    right: usize,
    anti: bool,
}
struct Edge {
    a: usize,
    b: usize,
    overlaps: Vec<Intersection>,
}
impl Edge {
    fn compatible(&self, a: &Candidate, b: &Candidate) -> bool {
        // Swapping two anticommuting data couplings induces a CZ between their
        // controls. Across the whole overlap these phases must cancel in pairs.
        let mut cross = false;
        for i in &self.overlaps {
            let left = a.slots[i.left];
            let right = b.slots[i.right];
            if left == right {
                return false;
            }
            cross ^= i.anti && left > right;
        }
        !cross
    }
}

impl Clock {
    pub(super) fn solve<'a>(
        &self,
        stars: &[Star],
        pipes: &[Pipe],
        cubes: impl Iterator<Item = (Basis, &'a [usize])>,
    ) -> Option<(Vec<usize>, Vec<Cat>)> {
        let cubes: Vec<_> = cubes.collect();
        for flexible in [false, true] {
            let mut checks: Vec<Check> = cubes
                .iter()
                .map(|&(b, ids)| Check {
                    events: ids.iter().map(|&id| (id, b)).collect(),
                    choices: choices(Shape::Cube {
                        width: ids.len(),
                        flexible,
                    }),
                })
                .collect();
            for star in stars {
                let (pipe, basis) = star.pipes[0];
                let widths = star.arms.iter().map(|a| a.events.len()).collect();
                // A hook on this checkerboard color can align with a logical string.
                let protected = star.arms.len() == 2
                    && basis == pipes[pipe].key.boundary_basis
                    && star.arms.iter().all(|a| a.data.len() == 2);
                checks.push(Check {
                    events: star
                        .arms
                        .iter()
                        .flat_map(|arm| arm.events.iter().map(|&id| (id, arm.basis)))
                        .collect(),
                    choices: choices(Shape::Wall {
                        widths,
                        protected,
                        axis: pipes[pipe].key.axis,
                    }),
                });
            }
            if let Some(solution) = solve_checks(self, &checks, cubes.len()) {
                let mut times = vec![0; self.data.len()];
                let mut cats = Vec::new();
                for (check, index) in checks.iter().zip(solution) {
                    let candidate = &check.choices[index];
                    for ((id, _), t) in check.events.iter().zip(&candidate.slots) {
                        times[*id] = *t as usize;
                    }
                    if let Some(cat) = &candidate.cat {
                        cats.push(cat.clone());
                    }
                }
                return Some((times, cats));
            }
        }
        None
    }
}

fn solve_checks(clock: &Clock, checks: &[Check], ordinary: usize) -> Option<Vec<usize>> {
    let mut data = FxMap::<IVec2, Vec<(usize, usize, Basis)>>::default();
    let mut pairs = BTreeMap::<(usize, usize), Vec<Intersection>>::new();
    for (b, check) in checks.iter().enumerate() {
        for (right, &(id, kind)) in check.events.iter().enumerate() {
            let seen = data.entry(clock.data[id]).or_default();
            for &(a, left, basis) in seen.iter() {
                pairs.entry((a, b)).or_default().push(Intersection {
                    left,
                    right,
                    anti: kind != basis,
                });
            }
            seen.push((b, right, kind));
        }
    }
    let edges: Vec<_> = pairs
        .into_iter()
        .map(|((a, b), overlaps)| Edge { a, b, overlaps })
        .collect();
    let mut incident = vec![Vec::new(); checks.len()];
    for (i, e) in edges.iter().enumerate() {
        incident[e.a].push(i);
        incident[e.b].push(i);
    }
    let mut domains: Vec<Vec<usize>> = checks
        .iter()
        .map(|v| (0..v.choices.len()).collect())
        .collect();
    if domains.iter().any(Vec::is_empty) {
        return None;
    }
    // Wall checks first, then the smallest remaining domain. Maintain the index
    // incrementally so distant ordinary tiles cost logarithmic time per choice.
    let mut unresolved: BTreeSet<_> = domains
        .iter()
        .enumerate()
        .filter(|(_, d)| d.len() > 1)
        .map(|(i, d)| (i < ordinary, d.len(), i))
        .collect();
    let change = |index: &mut BTreeSet<_>, q: usize, old: usize, new: usize| {
        if old > 1 {
            index.remove(&(q < ordinary, old, q));
        }
        if new > 1 {
            index.insert((q < ordinary, new, q));
        }
    };
    // Give the ordered templates a small deterministic search budget before
    // solving the same compatibility constraints with clause learning.
    let mut backtracks: u8 = 32;
    let mut trail = Vec::<(usize, Vec<usize>)>::new();
    let mut pending: VecDeque<_> = (0..edges.len()).collect();
    // Keep choice points and an undo trail on the heap even for large patches.
    let mut decisions = Vec::<(usize, usize, usize)>::new();
    loop {
        let mut valid = true;
        while let Some(i) = pending.pop_front() {
            let e = &edges[i];
            for (q, r) in [(e.a, e.b), (e.b, e.a)] {
                let next: Vec<_> = domains[q]
                    .iter()
                    .copied()
                    .filter(|&v| {
                        domains[r].iter().any(|&w| {
                            let (a, b) = if q == e.a { (v, w) } else { (w, v) };
                            e.compatible(&checks[e.a].choices[a], &checks[e.b].choices[b])
                        })
                    })
                    .collect();
                if next.len() == domains[q].len() {
                    continue;
                }
                if next.is_empty() {
                    valid = false;
                    break;
                }
                change(&mut unresolved, q, domains[q].len(), next.len());
                trail.push((q, std::mem::replace(&mut domains[q], next)));
                pending.extend(incident[q].iter().copied().filter(|&j| j != i));
            }
            if !valid {
                break;
            }
        }
        if valid {
            let Some(&(_, _, q)) = unresolved.first() else {
                return Some(domains.iter().map(|v| v[0]).collect());
            };
            decisions.push((q, 0, trail.len()));
        }
        pending.clear();
        loop {
            let (q, index, start) = decisions.last_mut()?;
            while trail.len() > *start {
                let (i, old) = trail.pop().expect("trail extends past this choice point");
                change(&mut unresolved, i, domains[i].len(), old.len());
                domains[i] = old;
            }
            if let Some(&chosen) = domains[*q].get(*index) {
                *index += 1;
                change(&mut unresolved, *q, domains[*q].len(), 1);
                trail.push((*q, std::mem::replace(&mut domains[*q], vec![chosen])));
                pending.extend(incident[*q].iter().copied());
                break;
            }
            decisions.pop();
            if backtracks == 0 {
                return solve_intersections(checks, &edges);
            }
            backtracks -= 1;
        }
    }
}

/// Learn conflicts across distant junctions when local propagation alone stalls.
/// On an edge, candidates with the same overlap slots impose identical constraints.
fn solve_intersections(checks: &[Check], edges: &[Edge]) -> Option<Vec<usize>> {
    use batsat::{BasicSolver, Lit, SolverInterface, lbool};
    let mut solver = BasicSolver::default();
    let mut selectors = Vec::with_capacity(checks.len());
    for check in checks {
        let candidates = (0..check.choices.len())
            .map(|_| Lit::new(solver.new_var_default(), true))
            .collect::<Vec<_>>();
        solver.add_clause_reuse(&mut candidates.clone());
        selectors.push(candidates);
    }
    for edge in edges {
        let mut classes = |q: usize| {
            let mut groups = BTreeMap::<Vec<u8>, (Lit, usize)>::new();
            for (i, c) in checks[q].choices.iter().enumerate() {
                let signature = edge
                    .overlaps
                    .iter()
                    .map(|x| c.slots[if q == edge.a { x.left } else { x.right }])
                    .collect();
                let (class, _) = *groups
                    .entry(signature)
                    .or_insert_with(|| (Lit::new(solver.new_var_default(), true), i));
                solver.add_clause_reuse(&mut vec![!selectors[q][i], class]);
            }
            groups.into_values().collect::<Vec<_>>()
        };
        let left = classes(edge.a);
        let right = classes(edge.b);
        for (a, i) in left {
            for &(b, j) in &right {
                if !edge.compatible(&checks[edge.a].choices[i], &checks[edge.b].choices[j]) {
                    solver.add_clause_reuse(&mut vec![!a, !b]);
                }
            }
        }
    }
    (solver.solve_limited(&[]) == lbool::TRUE).then(|| {
        selectors
            .iter()
            .map(|choices| {
                choices
                    .iter()
                    .position(|&lit| solver.value_lit(lit) == lbool::TRUE)
                    .expect("every check has a selected schedule")
            })
            .collect()
    })
}
