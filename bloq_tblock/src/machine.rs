//! Lattice-surgery back end: columns on a plane, in-plane spider trees between them.
//!
//! A layer has two wall families. Z columns meet in-plane X hubs, so an X tree
//! measures a Z parity or fans several CNOT controls into one target. X columns
//! meet in-plane Z hubs, so a Z tree fans one control out to several targets.
//! A link cell joins a column of one family to a tree of the other one layer up.

use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};

use bloq_graph::{
    Action, BinaryOp, Block, BlockGraph, BlockKind, CubeKind, Direction, Expr, FeedbackTarget,
    MeasureTarget, PauliBasis, Pipe, PortRole, SelectiveKind,
};
use glam::{IVec2, IVec3};

use crate::program::{Bits, Op, PhaseTerm, Program};

/// Failed layers before a stuck placement widens the layout.
const GROW_AFTER: i32 = 24;
/// Longest BFS path to a T site.
const SITE_REACH: usize = 8;
/// Layers held above a kept T site for its internal checks.
const HOLD: i32 = 3;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Color {
    Z,
    X,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Frame {
    /// Z merges along x, X merges along y.
    F0,
    /// Z merges along y, X merges along x.
    F1,
}

impl Frame {
    fn flipped(self) -> Self {
        match self {
            Self::F0 => Self::F1,
            Self::F1 => Self::F0,
        }
    }

    /// The in-plane axis of a Z face (`true`) or X face, as a unit step.
    fn face(self, z_face: bool) -> IVec2 {
        match (self, z_face) {
            (Self::F0, true) | (Self::F1, false) => IVec2::X,
            (Self::F1, true) | (Self::F0, false) => IVec2::Y,
        }
    }
}

fn perp(axis: IVec2) -> IVec2 {
    IVec2::new(axis.y.abs(), axis.x.abs())
}

#[derive(Clone, Copy, Debug)]
enum Cell {
    /// Column-type cube: colour plus a frame hint for cells without in-plane legs.
    Node {
        color: Color,
        frame: Frame,
    },
    /// In-plane spider between columns of the opposite colour.
    Hub {
        color: Color,
    },
    T,
    Cap,
    Y,
    Port(PortRole),
}

#[derive(Debug)]
struct Col {
    pos: IVec2,
    start: Frame,
    frame: Frame,
    /// First layer the next operation may use.
    floor: i32,
    /// Last layer touched.
    last: i32,
    /// Boundaries `z | z + 1` that carry a Hadamard.
    hadamards: BTreeSet<i32>,
    x_layers: BTreeSet<i32>,
    z_layers: BTreeSet<i32>,
}

/// What ends a parity hub.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Leaf {
    T { dagger: bool },
    Y { dagger: bool },
}

/// How phase batches use T sites.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TMode {
    /// One data hub and one selective cap per T.
    #[default]
    Gadget,
    /// One data hub per independent T parity; dependent T sites join internal checks.
    Block,
}

/// Column placement on the plane.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Placement {
    /// One bus row between two half-filled rows.
    Strip,
    /// Columns on odd coordinates, `columns` per row, with free cells around each.
    Grid {
        /// Columns per row; zero means a near-square grid.
        columns: usize,
    },
}

impl Default for Placement {
    fn default() -> Self {
        Self::Grid { columns: 0 }
    }
}

/// Back-end tuning.
#[derive(Clone, Copy, Debug, Default)]
pub struct MachineOptions {
    /// Graph layers reserved under every T block for its preparation lead.
    pub t_lead: i32,
    /// T site use.
    pub t_mode: TMode,
    /// Column placement.
    pub placement: Placement,
    /// Largest internal check, counting the dependent site.
    pub max_check: usize,
}

#[derive(Clone)]
struct Tree {
    cells: Vec<IVec2>,
    edges: Vec<(IVec2, IVec2)>,
}

/// An open T site, kept alive for internal checks.
#[derive(Clone, Debug)]
struct OpenSite {
    pos: IVec2,
    axis: IVec2,
    frame: Frame,
    /// Highest layer with a column cell.
    top: i32,
    selector: Vec<String>,
    dagger: bool,
}

pub(crate) struct Machine {
    cells: HashMap<IVec3, Cell>,
    pipes: Vec<(IVec3, Direction, bool)>,
    reserved: HashSet<IVec3>,
    /// Cells held for an open site's growth.
    owned: HashMap<IVec3, usize>,
    /// Open sites whose held cells count as free.
    allow: Vec<usize>,
    sites: Vec<OpenSite>,
    cols: Vec<Col>,
    is_col: HashSet<IVec2>,
    actions: Vec<Action>,
    /// Usable plane: `lo <= p < hi`.
    lo: IVec2,
    hi: IVec2,
    grown: usize,
    options: MachineOptions,
    names: usize,
    /// Counters for reports.
    pub(crate) checks: usize,
}

fn dir(from: IVec3, to: IVec3) -> Direction {
    Direction::try_from(to - from).expect("adjacent cells")
}

fn xor(names: &[String]) -> Expr {
    let mut iter = names.iter().map(|n| Expr::Var(n.clone()));
    let first = iter.next().expect("nonempty XOR");
    iter.fold(first, |a, b| {
        Expr::Binary(BinaryOp::Xor, Box::new(a), Box::new(b))
    })
}

impl Machine {
    pub(crate) fn new(order: &[usize], options: MachineOptions) -> Self {
        let k = order.len();
        let mut cols = Vec::with_capacity(k);
        cols.resize_with(k, || Col {
            pos: IVec2::ZERO,
            start: Frame::F1,
            frame: Frame::F1,
            floor: 1,
            last: 0,
            hadamards: BTreeSet::new(),
            x_layers: BTreeSet::new(),
            z_layers: BTreeSet::new(),
        });
        let mut is_col = HashSet::new();
        let (width, height);
        match options.placement {
            Placement::Strip => {
                for (slot, &col) in order.iter().enumerate() {
                    let pos = IVec2::new(slot as i32, if slot % 2 == 0 { 0 } else { 2 });
                    cols[col].pos = pos;
                    is_col.insert(pos);
                }
                width = k as i32;
                height = 3;
            }
            Placement::Grid { columns } => {
                let c = if columns == 0 {
                    (k as f64).sqrt().ceil() as usize
                } else {
                    columns
                }
                .max(1);
                let rows = k.div_ceil(c);
                for (slot, &col) in order.iter().enumerate() {
                    let pos = IVec2::new(2 * (slot % c) as i32 + 1, 2 * (slot / c) as i32 + 1);
                    cols[col].pos = pos;
                    is_col.insert(pos);
                }
                width = 2 * c as i32 + 1;
                height = 2 * rows as i32 + 1;
            }
        }
        Self {
            cells: HashMap::new(),
            pipes: Vec::new(),
            reserved: HashSet::new(),
            owned: HashMap::new(),
            allow: Vec::new(),
            sites: Vec::new(),
            cols,
            is_col,
            actions: Vec::new(),
            lo: IVec2::ZERO,
            hi: IVec2::new(width, height),
            grown: 0,
            options,
            names: 0,
            checks: 0,
        }
    }

    fn inside(&self, p: IVec2) -> bool {
        p.cmpge(self.lo).all() && p.cmplt(self.hi).all()
    }

    /// Add one free row or column, cycling around the plane.
    fn widen(&mut self) {
        match self.grown % 4 {
            0 => self.hi.x += 1,
            1 => self.lo.y -= 1,
            2 => self.hi.y += 1,
            _ => self.lo.x -= 1,
        }
        self.grown += 1;
    }

    fn free(&self, p: IVec3) -> bool {
        self.inside(p.truncate())
            && !self.is_col.contains(&p.truncate())
            && !self.cells.contains_key(&p)
            && !self.reserved.contains(&p)
            && self.owned.get(&p).is_none_or(|o| self.allow.contains(o))
    }

    fn node(&mut self, p: IVec3, color: Color, frame: Frame) {
        self.owned.remove(&p);
        let old = self.cells.insert(p, Cell::Node { color, frame });
        debug_assert!(old.is_none(), "cell {p} reused");
    }

    fn hub(&mut self, p: IVec3, color: Color) {
        self.owned.remove(&p);
        let old = self.cells.insert(p, Cell::Hub { color });
        debug_assert!(old.is_none(), "cell {p} reused");
    }

    fn pipe(&mut self, a: IVec3, b: IVec3) {
        self.pipes.push((a, dir(a, b), false));
    }

    fn fresh(&mut self, stem: &str) -> String {
        self.names += 1;
        format!("{stem}{}", self.names)
    }

    fn col_cell(&self, col: usize, z: i32) -> IVec3 {
        self.cols[col].pos.extend(z)
    }

    /// Free cells for a site's source, junction and cap around `p`.
    fn site_free(&self, p: IVec3, leaf: Leaf) -> bool {
        match leaf {
            Leaf::T { .. } => {
                let lead = self.options.t_lead;
                p.z - 1 - lead >= 0 && (-(1 + lead)..=1).all(|dz| self.free(p + IVec3::Z * dz))
            }
            Leaf::Y { .. } => self.free(p) && self.free(p + IVec3::Z),
        }
    }

    /// Face cells of a column in its current frame.
    fn faces(&self, col: usize, z_face: bool) -> [IVec2; 2] {
        let c = &self.cols[col];
        let axis = c.frame.face(z_face);
        [c.pos + axis, c.pos - axis]
    }

    /// Extend `tree` with a shortest free path to any goal. Returns the goal reached.
    fn grow(
        &self,
        z: i32,
        tree: &mut Tree,
        blocked: &HashSet<IVec2>,
        goals: &HashSet<IVec2>,
        reach: usize,
    ) -> Option<IVec2> {
        if let Some(&hit) = tree.cells.iter().find(|c| goals.contains(c)) {
            return Some(hit);
        }
        let mut parent: HashMap<IVec2, IVec2> = HashMap::new();
        let mut queue: VecDeque<(IVec2, usize)> = tree.cells.iter().map(|&c| (c, 0)).collect();
        let in_tree: HashSet<IVec2> = tree.cells.iter().copied().collect();
        while let Some((here, depth)) = queue.pop_front() {
            if depth >= reach {
                continue;
            }
            for step in [IVec2::X, IVec2::NEG_X, IVec2::Y, IVec2::NEG_Y] {
                let next = here + step;
                if in_tree.contains(&next)
                    || parent.contains_key(&next)
                    || blocked.contains(&next)
                    || !self.free(next.extend(z))
                {
                    continue;
                }
                parent.insert(next, here);
                if goals.contains(&next) {
                    let mut cell = next;
                    while let Some(&p) = parent.get(&cell) {
                        tree.cells.push(cell);
                        tree.edges.push((p, cell));
                        if in_tree.contains(&p) {
                            break;
                        }
                        cell = p;
                    }
                    return Some(next);
                }
                queue.push_back((next, depth + 1));
            }
        }
        None
    }

    /// A tree touching one face cell of every terminal. Returns the tree and entries.
    fn steiner(
        &self,
        z: i32,
        root: Option<IVec2>,
        terminals: &[Vec<IVec2>],
        blocked: &HashSet<IVec2>,
    ) -> Option<(Tree, Vec<IVec2>)> {
        let mut tree = Tree {
            cells: Vec::new(),
            edges: Vec::new(),
        };
        let mut entries: Vec<Option<IVec2>> = vec![None; terminals.len()];
        let usable = |c: &IVec2| !blocked.contains(c) && self.free(c.extend(z));
        if let Some(root) = root {
            if !usable(&root) {
                return None;
            }
            tree.cells.push(root);
        } else {
            let first = terminals.first()?.iter().copied().find(|c| usable(c))?;
            tree.cells.push(first);
            entries[0] = Some(first);
        }
        let reach = (self.hi - self.lo).element_sum() as usize * 2;
        loop {
            // Attach every terminal already touching the tree.
            for (i, faces) in terminals.iter().enumerate() {
                if entries[i].is_none()
                    && let Some(&hit) = faces.iter().find(|f| tree.cells.contains(f))
                {
                    entries[i] = Some(hit);
                }
            }
            let goals: HashSet<IVec2> = terminals
                .iter()
                .enumerate()
                .filter(|(i, _)| entries[*i].is_none())
                .flat_map(|(_, faces)| faces.iter().copied().filter(|c| usable(c)))
                .collect();
            if entries.iter().all(Option::is_some) {
                break;
            }
            if goals.is_empty() {
                return None;
            }
            self.grow(z, &mut tree, blocked, &goals, reach)?;
        }
        Some((tree, entries.into_iter().map(Option::unwrap).collect()))
    }

    fn lay_tree(&mut self, tree: &Tree, z: i32, color: Color) {
        for &c in &tree.cells {
            self.hub(c.extend(z), color);
        }
        for &(a, b) in &tree.edges {
            self.pipe(a.extend(z), b.extend(z));
        }
    }

    /// Plan a parity hub at layer `z` and commit it, or return `None`.
    fn try_hub(
        &mut self,
        cols: &[usize],
        leaf: Leaf,
        z: i32,
        keep: bool,
    ) -> Option<(String, usize)> {
        if cols
            .iter()
            .any(|&c| self.cols[c].x_layers.contains(&z) || self.cols[c].floor > z)
        {
            return None;
        }
        // A kept site needs an open far face for its later internal checks.
        let far_open = |m: &Self, site: IVec2, step: IVec2, tree: &[IVec2]| {
            let far = site + step;
            !keep || (m.free(far.extend(z)) && !tree.contains(&far))
        };
        if let [col] = *cols {
            for face in self.faces(col, true) {
                let pos = face.extend(z);
                let step = face - self.cols[col].pos;
                if self.site_free(pos, leaf) && far_open(self, face, step, &[]) {
                    let anchor = self.col_cell(col, z);
                    self.touch(col, z, Color::Z);
                    return Some(self.lay_site(anchor, pos, leaf, keep));
                }
            }
        }
        let terminals: Vec<Vec<IVec2>> =
            cols.iter().map(|&c| self.faces(c, true).to_vec()).collect();
        let (base, entries) = self.steiner(z, None, &terminals, &HashSet::new())?;
        // Reach a T site from the tree.
        let mut goals: HashSet<IVec2> = self.site_goals(z, leaf, &base, &HashSet::new());
        let (mut tree, site, link) = loop {
            let mut tree = base.clone();
            let site = self.grow(z, &mut tree, &HashSet::new(), &goals, SITE_REACH)?;
            // The site is the last cell the walk added; it is a column, not a hub.
            let (link, _) = *tree.edges.iter().find(|(_, b)| *b == site)?;
            if far_open(self, site, site - link, &tree.cells) {
                break (tree, site, link);
            }
            goals.remove(&site);
        };
        tree.cells.retain(|c| *c != site);
        tree.edges.retain(|(_, b)| *b != site);
        self.lay_tree(&tree, z, Color::X);
        for (&col, &entry) in cols.iter().zip(&entries) {
            self.touch(col, z, Color::Z);
            let cell = self.col_cell(col, z);
            self.pipe(entry.extend(z), cell);
        }
        Some(self.lay_site(link.extend(z), site.extend(z), leaf, keep))
    }

    fn site_goals(
        &self,
        z: i32,
        leaf: Leaf,
        tree: &Tree,
        blocked: &HashSet<IVec2>,
    ) -> HashSet<IVec2> {
        let mut goals = HashSet::new();
        let lo =
            tree.cells.iter().fold(IVec2::MAX, |a, &b| a.min(b)) - IVec2::splat(SITE_REACH as i32);
        let hi =
            tree.cells.iter().fold(IVec2::MIN, |a, &b| a.max(b)) + IVec2::splat(SITE_REACH as i32);
        for x in lo.x.max(self.lo.x)..=hi.x.min(self.hi.x - 1) {
            for y in lo.y.max(self.lo.y)..=hi.y.min(self.hi.y - 1) {
                let p = IVec2::new(x, y);
                if !tree.cells.contains(&p)
                    && !blocked.contains(&p)
                    && self.site_free(p.extend(z), leaf)
                {
                    goals.insert(p);
                }
            }
        }
        goals
    }

    /// Lay a site column at `site`, joined to `anchor` in the same layer.
    fn lay_site(&mut self, anchor: IVec3, site: IVec3, leaf: Leaf, keep: bool) -> (String, usize) {
        self.node(site, Color::Z, Frame::F1);
        self.pipes.push((anchor, dir(anchor, site), false));
        match leaf {
            Leaf::T { dagger } => {
                let source = site - IVec3::Z;
                self.cells.insert(source, Cell::T);
                self.pipe(source, site);
                for dz in 2..=1 + self.options.t_lead {
                    self.reserved.insert(site - IVec3::Z * dz);
                }
                let name = self.fresh("t");
                self.actions.push(Action::Measure {
                    target: MeasureTarget::Edge {
                        src: anchor,
                        dir: dir(anchor, site),
                    },
                    name: name.clone(),
                });
                let id = self.sites.len();
                let axis = (site - anchor).truncate().abs();
                self.sites.push(OpenSite {
                    pos: site.truncate(),
                    axis,
                    frame: if axis == IVec2::X {
                        Frame::F0
                    } else {
                        Frame::F1
                    },
                    top: site.z,
                    selector: vec![name.clone()],
                    dagger,
                });
                if keep {
                    // Hold the column and the far face; checks extend the column upwards.
                    let far = site + (site - anchor);
                    for dz in 1..=HOLD {
                        self.owned.entry(site + IVec3::Z * dz).or_insert(id);
                    }
                    for dz in 0..HOLD {
                        if self.free(far + IVec3::Z * dz) {
                            self.owned.insert(far + IVec3::Z * dz, id);
                        }
                    }
                } else {
                    self.close_site(id);
                }
                (name, id)
            }
            Leaf::Y { dagger } => {
                let cap = site + IVec3::Z;
                self.cells.insert(cap, Cell::Y);
                self.pipe(site, cap);
                if !dagger {
                    self.actions.push(Action::Feedback {
                        targets: vec![FeedbackTarget {
                            pauli: PauliBasis::Z,
                            target: site,
                            direction: Some(Direction::ZPLUS),
                        }],
                        condition: None,
                    });
                }
                (String::new(), usize::MAX)
            }
        }
    }

    /// Cap an open site with its selector.
    fn close_site(&mut self, id: usize) {
        let site = self.sites[id].clone();
        let cap = site.pos.extend(site.top + 1);
        self.owned.retain(|_, o| *o != id);
        self.cells.insert(cap, Cell::Cap);
        self.pipe(cap - IVec3::Z, cap);
        let c = xor(&site.selector);
        self.actions.push(Action::Resolve {
            target: cap,
            // T: X when the site matches its parity. T dagger: the opposite.
            condition: if site.dagger {
                c
            } else {
                Expr::Not(Box::new(c))
            },
        });
    }

    /// Grow an open site's column to `z`, if the cells are free.
    fn extend_site(&mut self, id: usize, z: i32) -> bool {
        let site = &self.sites[id];
        let (pos, top) = (site.pos, site.top);
        let ok = |m: &Self, p: IVec3| m.free(p) || m.owned.get(&p) == Some(&id);
        debug_assert!(self.allow.contains(&id) || self.allow.is_empty());
        if !((top + 1)..=(z + 1)).all(|dz| ok(self, pos.extend(dz))) {
            return false;
        }
        let frame = site.frame;
        for dz in (top + 1)..=z {
            let p = pos.extend(dz);
            self.owned.remove(&p);
            self.node(p, Color::Z, frame);
            self.pipe(p - IVec3::Z, p);
        }
        self.owned.insert(pos.extend(z + 1), id);
        self.sites[id].top = z;
        true
    }

    fn touch(&mut self, col: usize, z: i32, color: Color) {
        let c = &mut self.cols[col];
        c.last = c.last.max(z);
        match color {
            Color::Z => c.z_layers.insert(z),
            Color::X => c.x_layers.insert(z),
        };
    }

    fn z_feedback(&mut self, cols: &[usize]) {
        for &col in cols {
            // Each pipe end has its own frame: stay after an earlier Hadamard on this
            // pipe and before a later one.
            let last = self.cols[col].last;
            let (target, direction) = if self.cols[col].hadamards.contains(&last) {
                self.cols[col].last = last + 1;
                (self.col_cell(col, last + 1), Direction::ZMINUS)
            } else {
                (self.col_cell(col, last), Direction::ZPLUS)
            };
            self.actions.push(Action::Feedback {
                targets: vec![FeedbackTarget {
                    pauli: PauliBasis::Z,
                    target,
                    direction: Some(direction),
                }],
                condition: None,
            });
        }
    }

    fn leaf(quarters: u8) -> Option<Leaf> {
        match quarters {
            1 | 5 => Some(Leaf::T { dagger: false }),
            3 | 7 => Some(Leaf::T { dagger: true }),
            2 => Some(Leaf::Y { dagger: false }),
            6 => Some(Leaf::Y { dagger: true }),
            _ => None,
        }
    }

    fn span(&self, cols: &[usize]) -> i32 {
        let xs = cols.iter().map(|&c| self.cols[c].pos);
        let lo = xs.clone().fold(IVec2::MAX, IVec2::min);
        let hi = xs.fold(IVec2::MIN, IVec2::max);
        (hi - lo).element_sum()
    }

    fn phases(&mut self, terms: &[PhaseTerm]) {
        let mut order: Vec<&PhaseTerm> = terms.iter().collect();
        // First fit by decreasing span packs commuting hubs into few layers.
        order.sort_by_key(|t| std::cmp::Reverse(self.span(&t.cols)));
        if self.options.t_mode == TMode::Block {
            self.block(&order);
            return;
        }
        for term in order {
            self.gadget(term, false);
        }
    }

    /// One data hub with its own site. Returns the site's readout and id for a T.
    fn gadget(&mut self, term: &PhaseTerm, keep: bool) -> Option<(String, usize)> {
        let q = term.quarters;
        let Some(leaf) = Self::leaf(q) else {
            self.z_feedback(&term.cols);
            return None;
        };
        let mut z = term
            .cols
            .iter()
            .map(|&c| self.cols[c].floor)
            .max()
            .expect("nonempty parity");
        let mut tries = 0;
        let placed = loop {
            if let Some(placed) = self.try_hub(&term.cols, leaf, z, keep) {
                break placed;
            }
            z += 1;
            tries += 1;
            if tries % GROW_AFTER == 0 {
                self.widen();
            }
            assert!(tries < 100_000, "no site for parity {:?}", term.cols);
        };
        if matches!(q, 3 | 5) {
            self.z_feedback(&term.cols);
        }
        matches!(leaf, Leaf::T { .. }).then_some(placed)
    }

    fn max_check(&self) -> usize {
        if self.options.max_check == 0 {
            5
        } else {
            self.options.max_check
        }
    }

    /// Independent T parities get data hubs; dependent ones join internal checks.
    fn block(&mut self, order: &[&PhaseTerm]) {
        let k = self.cols.len();
        // Gaussian elimination with combinations over chosen basis terms.
        let mut pivots: Vec<(usize, Bits, Vec<usize>)> = Vec::new();
        let mut basis: Vec<&PhaseTerm> = Vec::new();
        let mut dependent: Vec<(&PhaseTerm, Vec<usize>)> = Vec::new();
        let mut rest: Vec<&PhaseTerm> = Vec::new();
        let mut t_terms: Vec<&PhaseTerm> = order
            .iter()
            .copied()
            .filter(|t| t.quarters % 2 == 1)
            .collect();
        t_terms.sort_by_key(|t| t.cols.len());
        rest.extend(order.iter().copied().filter(|t| t.quarters % 2 == 0));
        for term in t_terms {
            let mut v = Bits::zeros(k);
            for &c in &term.cols {
                v.flip(c);
            }
            let mut combo: Vec<usize> = Vec::new();
            for (col, row, rc) in &pivots {
                if v.get(*col) {
                    v.xor(row);
                    for &b in rc {
                        if let Some(i) = combo.iter().position(|&x| x == b) {
                            combo.swap_remove(i);
                        } else {
                            combo.push(b);
                        }
                    }
                }
            }
            if let Some(&col) = v.ones().first() {
                let id = basis.len();
                basis.push(term);
                pivots.push((col, v, vec![id]));
            } else if combo.len() < self.max_check() {
                dependent.push((term, combo));
            } else {
                rest.push(term);
            }
        }
        if dependent.is_empty() {
            for term in basis.into_iter().chain(rest) {
                self.gadget(term, false);
            }
            return;
        }
        let mut used: Vec<bool> = vec![false; basis.len()];
        for (_, combo) in &dependent {
            for &b in combo {
                used[b] = true;
            }
        }
        let mut handles: Vec<Option<usize>> = vec![None; basis.len()];
        let mut pending = dependent;
        for (i, term) in basis.iter().enumerate() {
            handles[i] = self.gadget(term, used[i]).map(|(_, id)| id);
            // Check each dependent site as soon as its basis sites exist.
            let (ready, wait): (Vec<_>, Vec<_>) = pending
                .into_iter()
                .partition(|(_, combo)| combo.iter().all(|&b| b <= i));
            pending = wait;
            for (term, combo) in ready {
                let ids: Vec<usize> = combo
                    .iter()
                    .map(|&b| handles[b].expect("basis T site"))
                    .collect();
                if !self.check(term, &ids) {
                    self.gadget(term, false);
                }
            }
        }
        debug_assert!(
            pending.is_empty(),
            "dependent terms span placed basis sites"
        );
        for id in handles.into_iter().flatten() {
            if self.owned.values().any(|&o| o == id) {
                self.close_site(id);
            }
        }
        for term in rest {
            self.gadget(term, false);
        }
    }

    /// A dependent T site joined to open sites by one internal check.
    fn check(&mut self, term: &PhaseTerm, ids: &[usize]) -> bool {
        self.allow = ids.to_vec();
        let placed = self.try_check(term, ids);
        self.allow.clear();
        placed
    }

    fn try_check(&mut self, term: &PhaseTerm, ids: &[usize]) -> bool {
        let leaf = Self::leaf(term.quarters).expect("T term");
        if !matches!(leaf, Leaf::T { .. }) {
            return false;
        }
        let start = ids.iter().map(|&i| self.sites[i].top).max().unwrap_or(1);
        for z in start..start + 3 * GROW_AFTER {
            if !ids.iter().all(|&i| {
                let s = &self.sites[i];
                ((s.top + 1)..=(z + 1)).all(|dz| self.free(s.pos.extend(dz)))
            }) {
                continue;
            }
            // Faces of open sites along their axes.
            let terminals: Vec<Vec<IVec2>> = ids
                .iter()
                .map(|&i| {
                    let s = &self.sites[i];
                    vec![s.pos + s.axis, s.pos - s.axis]
                })
                .collect();
            let blocked: HashSet<IVec2> = ids.iter().map(|&i| self.sites[i].pos).collect();
            let Some((mut tree, entries)) = self.steiner(z, None, &terminals, &blocked) else {
                continue;
            };
            let goals = self.site_goals(z, leaf, &tree, &blocked);
            let Some(site) = self.grow(z, &mut tree, &blocked, &goals, SITE_REACH) else {
                continue;
            };
            let Some(&(link, _)) = tree.edges.iter().find(|(_, b)| *b == site) else {
                continue;
            };
            tree.cells.retain(|c| *c != site);
            tree.edges.retain(|(_, b)| *b != site);
            for &i in ids {
                let grown = self.extend_site(i, z);
                debug_assert!(grown, "checked free");
            }
            self.lay_tree(&tree, z, Color::X);
            for (&i, &entry) in ids.iter().zip(&entries) {
                let cell = self.sites[i].pos.extend(z);
                self.pipe(entry.extend(z), cell);
            }
            let (name, id) = self.lay_site(link.extend(z), site.extend(z), leaf, false);
            // c_i = m_i + sum of the joined sites' basis readouts.
            let mut selector = vec![name];
            for &i in ids {
                selector.push(self.sites[i].selector[0].clone());
            }
            // Recap with the combined selector.
            self.actions.pop();
            self.sites[id].selector = selector;
            let cap = self.sites[id].pos.extend(self.sites[id].top + 1);
            let c = xor(&self.sites[id].selector);
            self.actions.push(Action::Resolve {
                target: cap,
                condition: if self.sites[id].dagger {
                    c
                } else {
                    Expr::Not(Box::new(c))
                },
            });
            self.checks += 1;
            if matches!(term.quarters, 3 | 5) {
                self.z_feedback(&term.cols);
            }
            return true;
        }
        false
    }

    /// Next barrier layer for these columns.
    fn barrier(&self, cols: &[usize]) -> i32 {
        cols.iter()
            .map(|&c| self.cols[c].floor.max(self.cols[c].last + 1))
            .max()
            .expect("nonempty columns")
    }

    /// One control fans out to targets: CNOT, or CZ with Hadamard-wrapped targets.
    fn try_fan_out(&mut self, control: usize, targets: &[usize], hadamard: bool, z: i32) -> bool {
        if self.cols[control].x_layers.contains(&z)
            || targets
                .iter()
                .any(|&t| self.cols[t].last > z || self.cols[t].floor > z + 1)
        {
            return false;
        }
        // CZ wraps each target in Hadamards, so it lands on the current Z face.
        let landings: Vec<Vec<IVec2>> = targets
            .iter()
            .map(|&t| self.faces(t, hadamard).to_vec())
            .collect();
        for face in self.faces(control, true) {
            if !self.free(face.extend(z)) || !self.free(face.extend(z + 1)) {
                continue;
            }
            let a = (face - self.cols[control].pos).abs();
            let b = perp(a);
            // A single target beside the link joins it directly.
            if let [t] = *targets {
                let tc = &self.cols[t];
                let t_axis = tc.frame.face(hadamard);
                if [b, -b].iter().any(|&s| face + s == tc.pos) && t_axis == b {
                    self.lay_fan_out(control, face, None, &[(t, face)], hadamard, z);
                    return true;
                }
            }
            for root in [face + b, face - b] {
                let blocked: HashSet<IVec2> = [face].into_iter().collect();
                if let Some((tree, entries)) = self.steiner(z + 1, Some(root), &landings, &blocked)
                {
                    let lands: Vec<(usize, IVec2)> = targets.iter().copied().zip(entries).collect();
                    self.lay_fan_out(control, face, Some(tree), &lands, hadamard, z);
                    return true;
                }
            }
        }
        false
    }

    fn lay_fan_out(
        &mut self,
        control: usize,
        face: IVec2,
        tree: Option<Tree>,
        lands: &[(usize, IVec2)],
        hadamard: bool,
        z: i32,
    ) {
        self.touch(control, z, Color::Z);
        let low = face.extend(z);
        let link = face.extend(z + 1);
        self.node(low, Color::Z, Frame::F0);
        self.pipe(self.col_cell(control, z), low);
        self.pipes.push((low, Direction::ZPLUS, false));
        self.node(link, Color::X, Frame::F0);
        if let Some(tree) = &tree {
            self.lay_tree(tree, z + 1, Color::Z);
            self.pipe(link, tree.cells[0].extend(z + 1));
        }
        for &(t, entry) in lands {
            self.touch(t, z + 1, Color::X);
            let cell = self.col_cell(t, z + 1);
            self.pipe(entry.extend(z + 1), cell);
            if hadamard {
                for boundary in [z, z + 1] {
                    let c = &mut self.cols[t];
                    if !c.hadamards.remove(&boundary) {
                        c.hadamards.insert(boundary);
                    }
                }
            }
            self.cols[t].floor = self.cols[t].floor.max(z + 2);
            self.cols[t].last = self.cols[t].last.max(z + 1);
        }
        self.cols[control].floor = self.cols[control].floor.max(z + 1);
    }

    /// Controls fan into one target: an X tree at `z` over the controls' Z faces.
    fn try_fan_in(&mut self, controls: &[usize], target: usize, z: i32) -> bool {
        let tc = &self.cols[target];
        if tc.last > z
            || tc.floor > z + 1
            || controls.iter().any(|&c| self.cols[c].x_layers.contains(&z))
        {
            return false;
        }
        let t_axis = tc.frame.face(false);
        for land in self.faces(target, false) {
            if !self.free(land.extend(z)) || !self.free(land.extend(z + 1)) {
                continue;
            }
            let a = perp(t_axis);
            let mut terminals: Vec<Vec<IVec2>> = controls
                .iter()
                .map(|&c| self.faces(c, true).to_vec())
                .collect();
            terminals.push(vec![land + a, land - a]);
            let blocked: HashSet<IVec2> = [land].into_iter().collect();
            let Some((tree, entries)) = self.steiner(z, None, &terminals, &blocked) else {
                continue;
            };
            self.lay_tree(&tree, z, Color::X);
            for (&c, &entry) in controls.iter().zip(&entries) {
                self.touch(c, z, Color::Z);
                let cell = self.col_cell(c, z);
                self.pipe(entry.extend(z), cell);
                self.cols[c].floor = self.cols[c].floor.max(z + 1);
            }
            let joint = *entries.last().expect("link terminal");
            let low = land.extend(z);
            self.node(low, Color::Z, Frame::F0);
            self.pipe(joint.extend(z), low);
            self.pipes.push((low, Direction::ZPLUS, false));
            let high = land.extend(z + 1);
            self.node(high, Color::X, Frame::F0);
            self.touch(target, z + 1, Color::X);
            self.pipe(high, self.col_cell(target, z + 1));
            self.cols[target].floor = self.cols[target].floor.max(z + 2);
            return true;
        }
        false
    }

    fn fan_out(&mut self, control: usize, targets: &[usize], hadamard: bool) {
        let mut z = self.barrier(&[control]).max(self.barrier(targets) - 1);
        let mut tries = 0;
        while !self.try_fan_out(control, targets, hadamard, z) {
            z += 1;
            tries += 1;
            if tries % GROW_AFTER == 0 {
                if targets.len() > 1 {
                    // Split a crowded star.
                    let (a, b) = targets.split_at(targets.len() / 2);
                    self.fan_out(control, a, hadamard);
                    self.fan_out(control, b, hadamard);
                    return;
                }
                self.widen();
            }
            assert!(
                tries < 100_000,
                "no fan-out route for {control} -> {targets:?}"
            );
        }
    }

    fn fan_in(&mut self, controls: &[usize], target: usize) {
        if let [control] = *controls {
            self.fan_out(control, &[target], false);
            return;
        }
        let mut z = self.barrier(controls).max(self.barrier(&[target]) - 1);
        let mut tries = 0;
        while !self.try_fan_in(controls, target, z) {
            z += 1;
            tries += 1;
            if tries % GROW_AFTER == 0 {
                let (a, b) = controls.split_at(controls.len() / 2);
                self.fan_in(a, target);
                self.fan_in(b, target);
                return;
            }
        }
    }

    fn hadamard(&mut self, col: usize) {
        let c = &mut self.cols[col];
        if !c.hadamards.remove(&c.last) {
            c.hadamards.insert(c.last);
        }
        c.frame = c.frame.flipped();
        c.floor = c.floor.max(c.last + 1);
    }

    pub(crate) fn run(&mut self, program: &Program) {
        for op in &program.ops {
            match op {
                Op::Phases(terms) => self.phases(terms),
                Op::FanOut { control, targets } => self.fan_out(*control, targets, false),
                Op::FanIn { controls, target } => self.fan_in(controls, *target),
                Op::CzStar { center, leaves } => self.fan_out(*center, leaves, true),
                Op::H { col } => self.hadamard(*col),
            }
        }
    }

    pub(crate) fn finish(mut self, program: &Program) -> Result<Built, String> {
        debug_assert!(self.owned.is_empty(), "open sites remain");
        let top = self
            .cells
            .keys()
            .map(|p| p.z)
            .chain(self.cols.iter().map(|c| c.last + 1))
            .max()
            .unwrap_or(1)
            .max(2);
        let mut inputs = Vec::with_capacity(self.cols.len());
        let mut tops = Vec::with_capacity(self.cols.len());
        for i in 0..self.cols.len() {
            let pos = self.cols[i].pos;
            let port = pos.extend(0);
            self.cells.insert(port, Cell::Port(PortRole::Input));
            let mut frame = self.cols[i].start;
            for z in 1..top {
                if self.cols[i].hadamards.contains(&(z - 1)) {
                    frame = frame.flipped();
                }
                let color = if self.cols[i].x_layers.contains(&z) {
                    Color::X
                } else {
                    Color::Z
                };
                self.node(pos.extend(z), color, frame);
            }
            for z in 0..top {
                let h = self.cols[i].hadamards.contains(&z);
                self.pipes.push((pos.extend(z), Direction::ZPLUS, h));
            }
            let out = pos.extend(top);
            self.cells.insert(out, Cell::Port(PortRole::Output));
            inputs.push(port);
            tops.push(out);
        }
        let outputs: Vec<IVec3> = program.outputs.iter().map(|&c| tops[c]).collect();
        for (wire, &col) in program.outputs.iter().enumerate() {
            // The output endpoint's frame is the wire's final frame, even across a Hadamard.
            let target = self.cols[col].pos.extend(top);
            let mut targets = Vec::new();
            for (flag, pauli) in [
                (program.x_frame[wire], PauliBasis::X),
                (program.z_frame[wire], PauliBasis::Z),
            ] {
                if flag {
                    targets.push(FeedbackTarget {
                        pauli,
                        target,
                        direction: Some(Direction::ZMINUS),
                    });
                }
            }
            if !targets.is_empty() {
                self.actions.push(Action::Feedback {
                    targets,
                    condition: None,
                });
            }
        }
        let graph = self.to_graph()?;
        Ok(Built {
            graph,
            inputs,
            outputs,
            checks: self.checks,
        })
    }

    fn to_graph(&self) -> Result<BlockGraph, String> {
        let mut axes: HashMap<IVec3, [bool; 3]> = HashMap::new();
        for &(src, d, _) in &self.pipes {
            let step = d.to_ivec3();
            let axis = if step.x != 0 {
                0
            } else if step.y != 0 {
                1
            } else {
                2
            };
            axes.entry(src).or_default()[axis] = true;
            axes.entry(src + step).or_default()[axis] = true;
        }
        let mut graph = BlockGraph::new();
        let mut sorted: Vec<(IVec3, Cell)> = self.cells.iter().map(|(p, c)| (*p, *c)).collect();
        sorted.sort_by_key(|(p, _)| (p.z, p.x, p.y));
        for (pos, cell) in sorted {
            let block = match cell {
                Cell::T => Block::new(pos, BlockKind::T),
                Cell::Cap => Block::new(pos, BlockKind::Selective(SelectiveKind::XY)),
                Cell::Y => Block::new(pos, BlockKind::Y),
                Cell::Port(role) => Block::new(pos, BlockKind::Port)
                    .with_port_role(role)
                    .map_err(|e| e.to_string())?,
                Cell::Hub { color } => Block::new(
                    pos,
                    BlockKind::Cube(match color {
                        Color::X => CubeKind::ZZX,
                        Color::Z => CubeKind::XXZ,
                    }),
                ),
                Cell::Node { color, frame } => {
                    let a = axes.get(&pos).copied().unwrap_or_default();
                    let kind = cube_kind(color, frame, a, pos).map_err(|e| {
                        let touching: Vec<_> = self
                            .pipes
                            .iter()
                            .filter(|(src, d, _)| *src == pos || *src + d.to_ivec3() == pos)
                            .collect();
                        format!("{e}: {touching:?}")
                    })?;
                    Block::new(pos, BlockKind::Cube(kind))
                }
            };
            graph.try_add_block(block).map_err(|e| e.to_string())?;
        }
        for &(src, d, hadamard) in &self.pipes {
            let pipe = Pipe::new(src, d);
            graph
                .try_add_pipe(if hadamard { pipe.with_hadamard() } else { pipe })
                .map_err(|e| format!("{src} {d:?}: {e}"))?;
        }
        graph
            .set_actions_deferred(self.actions.clone())
            .map_err(|e| e.to_string())?;
        Ok(graph)
    }
}

fn cube_kind(color: Color, frame: Frame, a: [bool; 3], pos: IVec3) -> Result<CubeKind, String> {
    use CubeKind::{XZX, XZZ, ZXX, ZXZ};
    Ok(match (color, a[0], a[1]) {
        (_, true, true) => return Err(format!("line cube at {pos} has two in-plane axes")),
        (Color::Z, true, false) => XZX,
        (Color::X, true, false) => ZXZ,
        (Color::Z, false, true) => ZXX,
        (Color::X, false, true) => XZZ,
        (Color::Z, false, false) => match frame {
            Frame::F0 => XZX,
            Frame::F1 => ZXX,
        },
        (Color::X, false, false) => match frame {
            Frame::F0 => XZZ,
            Frame::F1 => ZXZ,
        },
    })
}

/// A finished graph and its boundary, in wire order.
#[derive(Debug)]
pub struct Built {
    /// The block graph with actions.
    pub graph: BlockGraph,
    /// Input ports per wire.
    pub inputs: Vec<IVec3>,
    /// Output ports per wire.
    pub outputs: Vec<IVec3>,
    /// Internal checks used by T blocks.
    pub checks: usize,
}
