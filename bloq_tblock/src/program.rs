//! Deferred-Clifford front end.
//!
//! CNOT and CZ gates stay in software until a Hadamard needs its wire isolated.
//! Physical columns therefore hold a linear basis of the wire values, every
//! phase is a Z parity over those columns, and all pending phases commute. This
//! merges repeated parities and exposes the rank structure T blocks exploit.

use std::collections::BTreeMap;

use quizx::circuit::Circuit;
use quizx::gate::GType;

/// A column bit set.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Bits(Vec<u64>);

impl Bits {
    /// All zero with room for `n` bits.
    pub fn zeros(n: usize) -> Self {
        Self(vec![0; n.div_ceil(64).max(1)])
    }
    /// The unit vector `e_i`.
    pub fn unit(n: usize, i: usize) -> Self {
        let mut b = Self::zeros(n);
        b.flip(i);
        b
    }
    /// Bit `i`.
    pub fn get(&self, i: usize) -> bool {
        self.0[i / 64] >> (i % 64) & 1 == 1
    }
    /// Toggle bit `i`.
    pub fn flip(&mut self, i: usize) {
        self.0[i / 64] ^= 1 << (i % 64);
    }
    /// XOR `other` into this set.
    pub fn xor(&mut self, other: &Self) {
        for (a, b) in self.0.iter_mut().zip(&other.0) {
            *a ^= *b;
        }
    }
    /// Number of set bits.
    pub fn weight(&self) -> usize {
        self.0.iter().map(|w| w.count_ones() as usize).sum()
    }
    /// True when no bit is set.
    pub fn is_zero(&self) -> bool {
        self.0.iter().all(|w| *w == 0)
    }
    /// Set bits in increasing order.
    pub fn ones(&self) -> Vec<usize> {
        let mut out = Vec::with_capacity(self.weight());
        for (i, &word) in self.0.iter().enumerate() {
            let mut w = word;
            while w != 0 {
                out.push(i * 64 + w.trailing_zeros() as usize);
                w &= w - 1;
            }
        }
        out
    }
}

/// A Z-parity phase of `quarters * pi/4`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PhaseTerm {
    /// Physical columns in the parity.
    pub cols: Vec<usize>,
    /// Phase in units of pi/4, in `1..8`.
    pub quarters: u8,
}

/// One physical operation on columns.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Op {
    /// Mutually commuting phases; the back end may reorder them.
    Phases(Vec<PhaseTerm>),
    /// CNOT from `control` to every target.
    FanOut {
        /// Shared control.
        control: usize,
        /// Targets.
        targets: Vec<usize>,
    },
    /// CNOT from every control to `target`.
    FanIn {
        /// Controls.
        controls: Vec<usize>,
        /// Shared target.
        target: usize,
    },
    /// CZ between `center` and every leaf.
    CzStar {
        /// Shared qubit.
        center: usize,
        /// Other qubits.
        leaves: Vec<usize>,
    },
    /// Hadamard on one column.
    H {
        /// Column.
        col: usize,
    },
}

/// A column-level program.
#[derive(Clone, Debug, Default)]
pub struct Program {
    /// Column count, equal to the wire count.
    pub cols: usize,
    /// Operations in order.
    pub ops: Vec<Op>,
    /// Column holding each output wire.
    pub outputs: Vec<usize>,
    /// Final X frame per wire.
    pub x_frame: Vec<bool>,
    /// Final Z frame per wire.
    pub z_frame: Vec<bool>,
}

impl Program {
    /// T sites, counting odd phases.
    pub fn t_count(&self) -> usize {
        self.terms().filter(|t| t.quarters % 2 == 1).count()
    }

    fn terms(&self) -> impl Iterator<Item = &PhaseTerm> {
        self.ops.iter().flat_map(|op| match op {
            Op::Phases(terms) => terms.as_slice(),
            _ => &[],
        })
    }

    /// `(terms, sites, rank)` for each phase batch with at least one T site.
    pub fn batches(&self) -> Vec<(usize, usize, usize)> {
        let mut out = Vec::new();
        for op in &self.ops {
            if let Op::Phases(terms) = op {
                let sites: Vec<_> = terms.iter().filter(|t| t.quarters % 2 == 1).collect();
                if !sites.is_empty() {
                    let rows = sites
                        .iter()
                        .map(|t| {
                            let mut b = Bits::zeros(self.cols);
                            for &c in &t.cols {
                                b.flip(c);
                            }
                            b
                        })
                        .collect();
                    out.push((terms.len(), sites.len(), rank(rows)));
                }
            }
        }
        out
    }
}

/// GF(2) rank of the rows.
pub fn rank(mut rows: Vec<Bits>) -> usize {
    let mut r = 0;
    let bits = rows.first().map_or(0, |row| row.0.len() * 64);
    for c in 0..bits {
        let Some(p) = (r..rows.len()).find(|&i| rows[i].get(c)) else {
            continue;
        };
        rows.swap(r, p);
        let pivot = rows[r].clone();
        for (i, row) in rows.iter_mut().enumerate() {
            if i != r && row.get(c) {
                row.xor(&pivot);
            }
        }
        r += 1;
    }
    r
}

/// How much Clifford work the front end defers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Deferral {
    /// Keep CNOT and CZ in software until a Hadamard needs them.
    #[default]
    Full,
    /// Execute each CNOT and CZ where the source has it. An ablation baseline.
    None,
}

struct Tracker {
    k: usize,
    /// Wire values as parities over columns.
    wires: Vec<Bits>,
    /// Pending phases over columns, in first-use order.
    phases: BTreeMap<Bits, (usize, i32)>,
    next_phase: usize,
    /// Pending CZ edges over columns.
    cz: Vec<Bits>,
    x_frame: Vec<bool>,
    z_frame: Vec<bool>,
    ops: Vec<Op>,
    deferral: Deferral,
}

impl Tracker {
    fn add_phase(&mut self, parity: Bits, quarters: i32) {
        if parity.is_zero() {
            return;
        }
        let next = self.next_phase;
        let entry = self.phases.entry(parity).or_insert_with(|| (next, 0));
        if entry.0 == next {
            self.next_phase += 1;
        }
        entry.1 = (entry.1 + quarters).rem_euclid(8);
    }

    fn wire_phase(&mut self, wire: usize, mut quarters: i32) {
        if self.x_frame[wire] {
            // T X is proportional to X T^dagger.
            quarters = -quarters;
        }
        let parity = self.wires[wire].clone();
        self.add_phase(parity, quarters);
    }

    /// Record CZ(a, b) over parities of columns.
    fn add_cz(&mut self, a: &Bits, b: &Bits) {
        let (ones_a, ones_b) = (a.ones(), b.ones());
        for &i in &ones_a {
            for &j in &ones_b {
                if i == j {
                    self.add_phase(Bits::unit(self.k, i), 4);
                } else {
                    // Each unordered pair toggles once per ordered occurrence.
                    self.cz[i].flip(j);
                    self.cz[j].flip(i);
                }
            }
        }
    }

    /// Physical CNOT(control, target): rewrite every pending expression.
    fn cnot_basis(&mut self, control: usize, target: usize) {
        // v_target = v'_target + v_control, so a term with bit `target` gains `control`.
        let rewrite = |b: &mut Bits| {
            if b.get(target) {
                b.flip(control);
            }
        };
        self.wires.iter_mut().for_each(rewrite);
        let old = std::mem::take(&mut self.phases);
        for (mut parity, (order, quarters)) in old {
            rewrite(&mut parity);
            // Two rewritten parities can coincide only if the map were singular.
            self.phases.insert(parity, (order, quarters));
        }
        // Q'(v') = Q(E v'): neighbours of `target` gain the edge to `control`.
        let neighbours = self.cz[target].ones();
        for i in neighbours {
            if i == control {
                self.add_phase(Bits::unit(self.k, control), 4);
            } else {
                self.cz[i].flip(control);
                self.cz[control].flip(i);
            }
        }
    }

    fn emit_phases(&mut self, select: impl Fn(&Bits) -> bool) {
        let chosen: Vec<_> = self.phases.keys().filter(|p| select(p)).cloned().collect();
        let mut terms: Vec<(usize, PhaseTerm)> = Vec::new();
        for parity in chosen {
            let (order, quarters) = self.phases.remove(&parity).expect("selected key");
            if quarters != 0 {
                terms.push((
                    order,
                    PhaseTerm {
                        cols: parity.ones(),
                        quarters: quarters as u8,
                    },
                ));
            }
        }
        terms.sort_by_key(|t| t.0);
        if !terms.is_empty() {
            self.ops
                .push(Op::Phases(terms.into_iter().map(|t| t.1).collect()));
        }
    }

    fn fan_out(&mut self, control: usize, targets: Vec<usize>) {
        if targets.is_empty() {
            return;
        }
        for &t in &targets {
            self.cnot_basis(control, t);
        }
        self.ops.push(Op::FanOut { control, targets });
    }

    fn fan_in(&mut self, controls: Vec<usize>, target: usize) {
        if controls.is_empty() {
            return;
        }
        for &c in &controls {
            self.cnot_basis(c, target);
        }
        self.ops.push(Op::FanIn { controls, target });
    }

    /// Columns whose sum is `e_wire`, solving A y = e_wire.
    fn preimage(&self, wire: usize) -> Bits {
        let k = self.k;
        let mut rows: Vec<(Bits, bool)> = self
            .wires
            .iter()
            .enumerate()
            .map(|(i, row)| (row.clone(), i == wire))
            .collect();
        let mut pivot_of_col = vec![None; k];
        let mut r = 0;
        #[expect(
            clippy::needless_range_loop,
            reason = "column index drives the elimination"
        )]
        for c in 0..k {
            let Some(p) = (r..k).find(|&i| rows[i].0.get(c)) else {
                continue;
            };
            rows.swap(r, p);
            let pivot = rows[r].clone();
            for (i, row) in rows.iter_mut().enumerate() {
                if i != r && row.0.get(c) {
                    row.0.xor(&pivot.0);
                    row.1 ^= pivot.1;
                }
            }
            pivot_of_col[c] = Some(r);
            r += 1;
        }
        let mut out = Bits::zeros(k);
        for (c, pivot) in pivot_of_col.iter().enumerate() {
            if pivot.is_some_and(|row| rows[row].1) {
                out.flip(c);
            }
        }
        out
    }

    /// Make `wire` equal to one column that no other wire reads.
    fn isolate(&mut self, wire: usize) -> usize {
        let preimage = self.preimage(wire);
        let row = self.wires[wire].clone();
        // |preimage ∩ row| is odd because the row applied to the preimage is 1.
        let j = preimage
            .ones()
            .into_iter()
            .filter(|&c| row.get(c))
            .min_by_key(|&c| {
                let targets = preimage.weight() - 1;
                let controls = row.weight() - 1;
                // Prefer a pivot with little pending interaction.
                (targets + controls, self.cz[c].weight(), c)
            })
            .expect("invertible wire map");
        // Column ops: col_j ^= col_s is CNOT(j -> s); col_s ^= col_j is CNOT(s -> j).
        let out: Vec<usize> = preimage.ones().into_iter().filter(|&c| c != j).collect();
        self.fan_out(j, out);
        let controls: Vec<usize> = self.wires[wire]
            .ones()
            .into_iter()
            .filter(|&c| c != j)
            .collect();
        self.fan_in(controls, j);
        j
    }

    fn hadamard(&mut self, wire: usize) {
        let col = self.isolate(wire);
        debug_assert_eq!(self.wires[wire].ones(), vec![col]);
        self.emit_phases(|p| p.get(col));
        let leaves = self.cz[col].ones();
        for &leaf in &leaves {
            self.cz[leaf].flip(col);
        }
        self.cz[col] = Bits::zeros(self.k);
        if !leaves.is_empty() {
            self.ops.push(Op::CzStar {
                center: col,
                leaves,
            });
        }
        self.ops.push(Op::H { col });
        let (x, z) = (self.x_frame[wire], self.z_frame[wire]);
        self.x_frame[wire] = z;
        self.z_frame[wire] = x;
    }

    fn cnot(&mut self, c: usize, t: usize) {
        if self.deferral == Deferral::None {
            // Keep the source's physical CNOT: wires stay single columns here.
            let (cc, tc) = (self.isolate(c), self.isolate(t));
            self.flush_diagonal_on(&[cc, tc]);
            self.fan_out(cc, vec![tc]);
        }
        let row = self.wires[c].clone();
        self.wires[t].xor(&row);
        self.x_frame[t] ^= self.x_frame[c];
        self.z_frame[c] ^= self.z_frame[t];
    }

    fn flush_diagonal_on(&mut self, cols: &[usize]) {
        let cols = cols.to_vec();
        self.emit_phases(|p| cols.iter().any(|&c| p.get(c)));
        for &c in &cols {
            let leaves = self.cz[c].ones();
            for &leaf in &leaves {
                self.cz[leaf].flip(c);
            }
            self.cz[c] = Bits::zeros(self.k);
            if !leaves.is_empty() {
                self.ops.push(Op::CzStar { center: c, leaves });
            }
        }
    }

    fn cz_gate(&mut self, a: usize, b: usize) {
        if self.x_frame[a] {
            self.z_frame[b] ^= true;
        }
        if self.x_frame[b] {
            self.z_frame[a] ^= true;
        }
        let (ra, rb) = (self.wires[a].clone(), self.wires[b].clone());
        self.add_cz(&ra, &rb);
    }

    /// Flush everything and reduce the wire map to a permutation.
    fn finish(mut self) -> Program {
        self.emit_phases(|_| true);
        loop {
            let center = (0..self.k).max_by_key(|&c| (self.cz[c].weight(), std::cmp::Reverse(c)));
            let Some(center) = center.filter(|&c| self.cz[c].weight() > 0) else {
                break;
            };
            let leaves = self.cz[center].ones();
            for &leaf in &leaves {
                self.cz[leaf].flip(center);
            }
            self.cz[center] = Bits::zeros(self.k);
            self.ops.push(Op::CzStar { center, leaves });
        }
        // Order wires cheapest first; each isolate fixes one wire and one column.
        let mut done = vec![false; self.k];
        for _ in 0..self.k {
            let wire = (0..self.k)
                .filter(|&w| !done[w])
                .min_by_key(|&w| {
                    let col_weight = self.preimage(w).weight();
                    (self.wires[w].weight() + col_weight, w)
                })
                .expect("a wire remains");
            self.isolate(wire);
            done[wire] = true;
        }
        let outputs = self
            .wires
            .iter()
            .map(|row| {
                let ones = row.ones();
                debug_assert_eq!(ones.len(), 1);
                ones[0]
            })
            .collect();
        Program {
            cols: self.k,
            ops: self.ops,
            outputs,
            x_frame: self.x_frame,
            z_frame: self.z_frame,
        }
    }
}

/// Lower a normalized Clifford+T circuit to a column program.
///
/// # Errors
///
/// Returns an error for a gate outside the static Clifford+T set.
pub fn lower(circuit: &Circuit, deferral: Deferral) -> Result<Program, String> {
    let k = circuit.num_qubits();
    let mut tracker = Tracker {
        k,
        wires: (0..k).map(|i| Bits::unit(k, i)).collect(),
        phases: BTreeMap::new(),
        next_phase: 0,
        cz: vec![Bits::zeros(k); k],
        x_frame: vec![false; k],
        z_frame: vec![false; k],
        ops: Vec::new(),
        deferral,
    };
    for gate in &circuit.gates {
        let q = &gate.qs;
        let quarters = || -> Result<i32, String> {
            let phase = gate.phase.to_rational();
            let (n, d) = (*phase.numer(), *phase.denom());
            if 4 % d != 0 {
                return Err(format!("non-Clifford+T phase {}", gate.phase));
            }
            i32::try_from(n * (4 / d)).map_err(|e| e.to_string())
        };
        match gate.t {
            GType::HAD => tracker.hadamard(q[0]),
            GType::NOT => tracker.x_frame[q[0]] ^= true,
            GType::Z => tracker.z_frame[q[0]] ^= true,
            GType::S => tracker.wire_phase(q[0], 2),
            GType::Sdg => tracker.wire_phase(q[0], -2),
            GType::T => tracker.wire_phase(q[0], 1),
            GType::Tdg => tracker.wire_phase(q[0], -1),
            GType::ZPhase => tracker.wire_phase(q[0], quarters()?),
            GType::XPhase => {
                let quarter = quarters()?;
                tracker.hadamard(q[0]);
                tracker.wire_phase(q[0], quarter);
                tracker.hadamard(q[0]);
            }
            GType::CNOT => tracker.cnot(q[0], q[1]),
            GType::CZ => tracker.cz_gate(q[0], q[1]),
            other => return Err(format!("gate {other:?} was not normalized")),
        }
    }
    Ok(tracker.finish())
}
