//! Parser half of the text exchange format. See the module doc in [`super`]
//! for the grammar; [`super::write`] is the writer half, and the two must
//! agree token for token.
//!
//! Hand-rolled and line-oriented: every statement is one line and blocks open
//! with a trailing `{` and close with a lone `}`, so no lookahead or
//! backtracking is ever needed.

use std::cell::Cell;
use std::collections::BTreeMap;
use std::fmt;
use std::str::FromStr;

use bloq_circuit::{
    BodyId, CircuitBody, ConditionalCorrection, CoordCircuit, DetectorParity, DetectorTerm, Flow,
    FlowMarker, GateType, LoopStateId, MeasRecord, Op, Pauli, PauliBasis, PauliMap,
    checked_translate_coordinate,
};
use glam::{IVec2, IVec3};
use rustc_hash::FxHashMap;

use crate::{
    Basis, Bloq, BloqEdge, BloqNode, BloqNodeId, BloqNodeKind, BloqTemplate, BloqTemplatePool,
    BoundaryFace, BundleDetector, BundleMeasurement, ClassicalExpr, ClassicalNode, DetectorBundle,
    DetectorBundleId, DetectorBundlePool, DetectorBundleUse, InstanceBoundaryOperator,
    InstanceMeasurement, MetadataValue, NodeDetector, NodeProvenance, NodeRestart,
    ObservableOutput, PipePadding, PipeSeam, QuantumEdge, QuantumNode, QuantumTimeline, RegionNode,
    SourceBlockRef, SubGraph, TemplateDetector, TemplateDetectorScope, TemplateId,
    TemplateInstance, TemplateInstanceId, TemplateRepeatState, TemplateRestart, TemporalPipeRef,
    ValueRef, ValueRole,
};

use super::{BLOQ_TEXT_VERSION, decode_metadata_token};

/// A syntax or structure error in Bloq IR text, with its 1-based line number.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("bloq text line {line}: {message}")]
pub struct TextParseError {
    /// One-based source line.
    pub line: usize,
    message: String,
}

impl TextParseError {
    fn new(line: usize, message: impl fmt::Display) -> Self {
        Self {
            line,
            message: message.to_string(),
        }
    }
}

type Result<T> = std::result::Result<T, TextParseError>;

pub(super) fn parse_bloq_text(text: &str) -> Result<Bloq> {
    Parser::new(text).parse_program()
}

/// The `(measurement id, qubit)` pairs a circuit's ops imply: one per
/// `Measure` target, one per `MPP` product (its representative coordinate).
/// The registry is reconstructed from these plus explicit `meas` lines, and
/// the writer emits a `meas` line exactly for the records not implied here.
pub(super) fn implied_measurement_qubits<'a>(
    ops: impl Iterator<Item = &'a Op>,
) -> FxHashMap<u32, IVec2> {
    let mut implied = FxHashMap::default();
    for op in ops {
        for_each_implied_measurement(op, |measurement, qubit| {
            implied.insert(measurement, qubit);
        });
    }
    implied
}

fn for_each_implied_measurement(op: &Op, mut f: impl FnMut(u32, IVec2)) {
    match op {
        Op::Measure {
            qubits,
            measurements,
            ..
        } => {
            for (&qubit, &measurement) in qubits.iter().zip(measurements) {
                f(measurement, qubit);
            }
        }
        Op::MPP {
            products,
            measurements,
        } => {
            for (product, &measurement) in products.iter().zip(measurements) {
                if let Some(qubit) = product.representative_coord() {
                    f(measurement, qubit);
                }
            }
        }
        Op::Gate { .. }
        | Op::Tick
        | Op::Repeat { .. }
        | Op::ConditionalPauli(_)
        | Op::Depolarize1 { .. }
        | Op::Depolarize2 { .. }
        | Op::PauliError { .. } => {}
    }
}

fn register_implied_measurements(
    op: &Op,
    line: usize,
    implied: &mut FxHashMap<u32, IVec2>,
) -> Result<()> {
    let mut conflict = None;
    for_each_implied_measurement(op, |measurement, qubit| match implied.get(&measurement) {
        Some(&previous) if previous != qubit => {
            conflict.get_or_insert((measurement, previous, qubit));
        }
        Some(_) => {}
        None => {
            implied.insert(measurement, qubit);
        }
    });
    let Some((measurement, first, second)) = conflict else {
        return Ok(());
    };
    Err(TextParseError::new(
        line,
        format!(
            "measurement m{measurement} is bound to two different qubits: \
             ({},{}) and ({},{})",
            first.x, first.y, second.x, second.y
        ),
    ))
}

/// Text is an untrusted boundary. Reject translations that would overflow the
/// `i32` lattice before later layout/flow code performs ordinary `IVec2`
/// addition (which panics in debug builds).
fn validate_text_coordinate_translations(bloq: &Bloq, line: usize) -> Result<()> {
    for (_, level) in bloq.levels() {
        for (_, quantum) in level.quantum_nodes() {
            for instance in &quantum.instances {
                validate_template_translation(bloq, instance.template_id, instance.offset, line)?;
            }
        }
    }
    for padding in bloq.pipe_padding() {
        for template in [padding.one_round, padding.looped] {
            validate_template_translation(bloq, template, padding.offset, line)?;
        }
    }
    Ok(())
}

fn validate_template_translation(
    bloq: &Bloq,
    template_id: TemplateId,
    offset: IVec2,
    line: usize,
) -> Result<()> {
    let Some(template) = bloq.templates().get(template_id) else {
        // Unknown template ids are diagnosed by semantic validation.
        return Ok(());
    };
    for &coord in template.qubits() {
        validate_translated_coord(coord, offset, template_id, line)?;
    }
    for flow in &template.boundary_flows {
        for (coord, _) in flow.start.iter().chain(flow.end.iter()) {
            validate_translated_coord(*coord, offset, template_id, line)?;
        }
        if let Some(center) = flow.center {
            validate_translated_coord(center, offset, template_id, line)?;
        }
    }
    Ok(())
}

fn validate_translated_coord(
    coord: IVec2,
    offset: IVec2,
    template: TemplateId,
    line: usize,
) -> Result<()> {
    if checked_translate_coordinate(coord, offset).is_err() {
        return Err(TextParseError::new(
            line,
            format!(
                "template t{} coordinate ({},{}) overflows after offset ({},{})",
                template.0, coord.x, coord.y, offset.x, offset.y
            ),
        ));
    }
    Ok(())
}

// ==============================================================================
// Line source
// ==============================================================================

struct Parser<'a> {
    /// Raw lines, split on `\n`.
    lines: Vec<&'a str>,
    /// Index of the next unread line.
    pos: usize,
}

/// One statement line, pre-split into whitespace tokens. The cursor is a
/// `Cell` so consuming a token needs only `&self` — several parsers pass the
/// statement to helpers while also advancing it in the same expression.
struct Stmt<'a> {
    line: usize,
    tokens: Vec<&'a str>,
    next: Cell<usize>,
}

impl<'a> Parser<'a> {
    fn new(text: &'a str) -> Self {
        Self {
            lines: text.split('\n').collect(),
            pos: 0,
        }
    }

    /// The next statement line — comments stripped, blank lines skipped.
    /// `None` at end of input.
    fn next_stmt(&mut self) -> Option<Stmt<'a>> {
        while self.pos < self.lines.len() {
            let line_number = self.pos + 1;
            let line = self.lines[self.pos];
            self.pos += 1;
            let code = line.split('#').next().unwrap_or("");
            let tokens: Vec<&str> = code.split_whitespace().collect();
            if !tokens.is_empty() {
                return Some(Stmt {
                    line: line_number,
                    tokens,
                    next: Cell::new(0),
                });
            }
        }
        None
    }

    fn expect_stmt(&mut self, expected: &str) -> Result<Stmt<'a>> {
        let end = self.lines.len();
        self.next_stmt().ok_or_else(|| {
            TextParseError::new(end, format!("expected {expected}, found end of input"))
        })
    }
}

impl<'a> Stmt<'a> {
    fn err(&self, message: impl fmt::Display) -> TextParseError {
        TextParseError::new(self.line, message)
    }

    fn peek(&self) -> Option<&'a str> {
        self.tokens.get(self.next.get()).copied()
    }

    fn advance(&self) {
        self.next.set(self.next.get() + 1);
    }

    fn next_token(&self, expected: &str) -> Result<&'a str> {
        let token = self
            .peek()
            .ok_or_else(|| self.err(format!("expected {expected}, found end of line")))?;
        self.advance();
        Ok(token)
    }

    fn expect(&self, literal: &str) -> Result<()> {
        let token = self.next_token(&format!("`{literal}`"))?;
        if token != literal {
            return Err(self.err(format!("expected `{literal}`, found `{token}`")));
        }
        Ok(())
    }

    fn finish(&self) -> Result<()> {
        if let Some(extra) = self.peek() {
            return Err(self.err(format!("unexpected trailing `{extra}`")));
        }
        Ok(())
    }

    /// Whether this statement is the lone `}` that closes a block.
    fn is_close(&self) -> bool {
        self.tokens == ["}"]
    }
}

// ==============================================================================
// Program structure
// ==============================================================================

impl<'a> Parser<'a> {
    fn parse_program(&mut self) -> Result<Bloq> {
        let header = self.expect_stmt("`BLOQIR <version>` header")?;
        header.expect("BLOQIR")?;
        let version: u32 = parse_int(&header, header.next_token("format version")?)?;
        header.finish()?;
        if version != BLOQ_TEXT_VERSION {
            return Err(header.err(format!(
                "unsupported bloq text version {version} (this reader supports only \
                 version {BLOQ_TEXT_VERSION}; there is no cross-version compatibility)"
            )));
        }

        let mut metadata = BTreeMap::new();
        let mut logical_inputs = Vec::new();
        let mut logical_outputs = Vec::new();
        let mut templates = BloqTemplatePool::new();
        let mut detector_bundles = DetectorBundlePool::new();
        let mut top = None;

        while let Some(stmt) = self.next_stmt() {
            match stmt.next_token(
                "`metadata`, `logical-input`, `logical-output`, `template`, `bundle`, or `graph`",
            )? {
                "metadata" => {
                    let key_token = stmt.next_token("metadata key")?;
                    let key = decode_metadata_token(key_token).map_err(|message| {
                        stmt.err(format!("invalid metadata key `{key_token}`: {message}"))
                    })?;
                    let kind = stmt.next_token("metadata value type")?;
                    let value_token = stmt.next_token("metadata value")?;
                    let value = match kind {
                        "u64" => MetadataValue::U64(parse_int(&stmt, value_token)?),
                        "string" => MetadataValue::String(
                            decode_metadata_token(value_token).map_err(|message| {
                                stmt.err(format!(
                                    "invalid metadata string `{value_token}`: {message}"
                                ))
                            })?,
                        ),
                        other => {
                            return Err(stmt.err(format!("unknown metadata value type `{other}`")));
                        }
                    };
                    stmt.finish()?;
                    if metadata.insert(key.clone(), value).is_some() {
                        return Err(stmt.err(format!("duplicate metadata key `{key}`")));
                    }
                }
                "logical-input" => {
                    let port = parse_coord3(&stmt, stmt.next_token("input port")?)?;
                    stmt.expect("instance")?;
                    let instance = crate::lowering::TemplateInstanceId(parse_prefixed(
                        &stmt,
                        stmt.next_token("input Port instance")?,
                        'i',
                    )?);
                    stmt.expect("x")?;
                    let x = parse_pauli_map(&stmt, stmt.next_token("logical X operator")?)?;
                    stmt.expect("z")?;
                    let z = parse_pauli_map(&stmt, stmt.next_token("logical Z operator")?)?;
                    stmt.finish()?;
                    logical_inputs.push(crate::LogicalInput {
                        port,
                        instance,
                        x,
                        z,
                    });
                }
                "logical-output" => {
                    let port = parse_coord3(&stmt, stmt.next_token("output port")?)?;
                    stmt.expect("instance")?;
                    let instance = crate::lowering::TemplateInstanceId(parse_prefixed(
                        &stmt,
                        stmt.next_token("output instance")?,
                        'i',
                    )?);
                    stmt.expect("x")?;
                    let x = parse_pauli_map(&stmt, stmt.next_token("logical X operator")?)?;
                    stmt.expect("z")?;
                    let z = parse_pauli_map(&stmt, stmt.next_token("logical Z operator")?)?;
                    stmt.finish()?;
                    logical_outputs.push(crate::LogicalOutput {
                        port,
                        instance,
                        x,
                        z,
                    });
                }
                "template" => {
                    let id = parse_prefixed(&stmt, stmt.next_token("template id")?, 't')?;
                    if id as usize != templates.len() {
                        return Err(stmt.err(format!(
                            "template t{id} out of order: expected t{}",
                            templates.len()
                        )));
                    }
                    stmt.expect("{")?;
                    stmt.finish()?;
                    templates.insert(self.parse_template()?);
                }
                "bundle" => {
                    let id = parse_prefixed(&stmt, stmt.next_token("bundle id")?, 'b')?;
                    if id as usize != detector_bundles.len() {
                        return Err(stmt.err(format!(
                            "bundle b{id} out of order: expected b{}",
                            detector_bundles.len()
                        )));
                    }
                    stmt.expect("owners")?;
                    let mut owner_templates = Vec::new();
                    while stmt.peek() != Some("{") {
                        owner_templates.push(TemplateId(parse_prefixed(
                            &stmt,
                            stmt.next_token("owner template or `{`")?,
                            't',
                        )?));
                    }
                    stmt.expect("{")?;
                    stmt.finish()?;
                    detector_bundles.insert(self.parse_bundle(owner_templates)?);
                }
                "graph" => {
                    if top.is_some() {
                        return Err(stmt.err("duplicate graph block"));
                    }
                    stmt.expect("{")?;
                    stmt.finish()?;
                    top = Some(self.parse_level()?);
                }
                other => return Err(stmt.err(format!("unknown section `{other}`"))),
            }
        }

        let top =
            top.ok_or_else(|| TextParseError::new(self.lines.len(), "missing `graph` section"))?;
        let bloq = Bloq::from_parts(
            templates,
            detector_bundles,
            top,
            metadata,
            logical_inputs,
            logical_outputs,
        );
        validate_text_coordinate_translations(&bloq, self.lines.len())?;
        Ok(bloq)
    }
}

impl<'a> Parser<'a> {
    fn parse_bundle(&mut self, owner_templates: Vec<TemplateId>) -> Result<DetectorBundle> {
        let mut detectors = Vec::new();
        loop {
            let stmt = self.expect_stmt("a bundle detector or `}`")?;
            if stmt.is_close() {
                break;
            }
            stmt.expect("detector")?;
            let parity = parse_parity(
                &stmt,
                stmt.next_token("bundle detector parity")?,
                parse_bundle_measurement,
            )?;
            let coords = parse_opt_detector_coords(&stmt)?;
            stmt.finish()?;
            detectors.push(BundleDetector { parity, coords });
        }
        Ok(DetectorBundle::new(owner_templates, detectors))
    }
}

// ==============================================================================
// Templates
// ==============================================================================

impl<'a> Parser<'a> {
    fn parse_template(&mut self) -> Result<BloqTemplate> {
        // Op lines per body id; bare `circuit` is shorthand for entry b0.
        let mut entry = None;
        let mut bodies: FxHashMap<u32, (Vec<Op>, usize)> = FxHashMap::default();
        // Explicit `meas` records with their line numbers for conflict errors.
        let mut extra_measurements: Vec<(MeasRecord, usize)> = Vec::new();
        let mut implied_measurements = FxHashMap::default();
        let mut detectors = Vec::new();
        let mut repeat_states = Vec::new();
        let mut boundary_flows = Vec::new();
        let mut restarts = Vec::new();

        loop {
            let stmt = self.expect_stmt("a template statement or `}`")?;
            if stmt.is_close() {
                break;
            }
            let keyword = stmt.next_token("template statement")?;
            match keyword {
                "circuit" | "body" => {
                    let body = if keyword == "circuit" && stmt.peek() == Some("{") {
                        0
                    } else {
                        parse_prefixed(&stmt, stmt.next_token("body id")?, 'b')?
                    };
                    if keyword == "circuit" && entry.replace(BodyId(body)).is_some() {
                        return Err(stmt.err("duplicate circuit entry"));
                    }
                    stmt.expect("{")?;
                    stmt.finish()?;
                    if bodies
                        .insert(
                            body,
                            (self.parse_ops(&mut implied_measurements)?, stmt.line),
                        )
                        .is_some()
                    {
                        return Err(stmt.err(format!("duplicate body b{body}")));
                    }
                }
                "meas" => {
                    let id = parse_measurement_output(&stmt, stmt.next_token("measurement id")?)?;
                    stmt.expect("@")?;
                    let qubit = parse_coord2(&stmt, stmt.next_token("qubit coordinate")?)?;
                    stmt.finish()?;
                    extra_measurements.push((MeasRecord { id, qubit }, stmt.line));
                }
                "detector" => {
                    let scope = if let Some(token) = stmt.peek()
                        && let Some(args) = call_args(token, "body")
                    {
                        stmt.advance();
                        let [body] = parse_call_args(&stmt, args)?;
                        TemplateDetectorScope::RepeatBody {
                            body: BodyId(parse_prefixed(&stmt, body, 'b')?),
                        }
                    } else {
                        TemplateDetectorScope::TopLevel
                    };
                    let parity =
                        parse_parity(&stmt, stmt.next_token("detector parity")?, parse_meas_term)?;
                    let coords = parse_opt_detector_coords(&stmt)?;
                    stmt.finish()?;
                    detectors.push(TemplateDetector {
                        scope,
                        parity,
                        coords,
                    });
                }
                token if call_args(token, "loop").is_some() => {
                    let args = call_args(token, "loop").expect("guard matched");
                    let [body, state] = parse_call_args(&stmt, args)?;
                    let body = BodyId(parse_prefixed(&stmt, body, 'b')?);
                    let state = LoopStateId(parse_prefixed(&stmt, state, 's')?);
                    stmt.expect("init")?;
                    let initial =
                        parse_parity(&stmt, stmt.next_token("initial parity")?, parse_meas_term)?;
                    stmt.expect("next")?;
                    let next =
                        parse_parity(&stmt, stmt.next_token("next parity")?, parse_meas_term)?;
                    stmt.finish()?;
                    repeat_states.push(TemplateRepeatState {
                        body,
                        state,
                        initial,
                        next,
                    });
                }
                "flow" => {
                    boundary_flows.push(parse_flow(&stmt)?);
                }
                "restart" => {
                    let parity =
                        parse_parity(&stmt, stmt.next_token("restart parity")?, parse_meas_term)?;
                    stmt.finish()?;
                    restarts.push(TemplateRestart { parity });
                }
                other => return Err(stmt.err(format!("unknown template statement `{other}`"))),
            }
        }

        Ok(BloqTemplate::with_parts(
            build_circuit(
                bodies,
                entry.unwrap_or(BodyId(0)),
                implied_measurements,
                extra_measurements,
            )?,
            detectors,
            repeat_states,
            boundary_flows,
            restarts,
        ))
    }

    fn parse_ops(&mut self, implied: &mut FxHashMap<u32, IVec2>) -> Result<Vec<Op>> {
        let mut ops = Vec::new();
        loop {
            let stmt = self.expect_stmt("a circuit op or `}`")?;
            if stmt.is_close() {
                return Ok(ops);
            }
            let op = parse_op(&stmt)?;
            register_implied_measurements(&op, stmt.line, implied)?;
            ops.push(op);
        }
    }
}

/// Assemble a [`CoordCircuit`] from per-body op lists and explicit registry
/// records. Body ids must be dense because [`CoordCircuit`] stores them in a
/// vector; the measurement registry is the ops' implied records plus the
/// explicit ones.
fn build_circuit(
    bodies: FxHashMap<u32, (Vec<Op>, usize)>,
    entry: BodyId,
    implied: FxHashMap<u32, IVec2>,
    extra: Vec<(MeasRecord, usize)>,
) -> Result<CoordCircuit> {
    if let Some((max_body, (_, line))) = bodies.iter().max_by_key(|(body, _)| **body)
        && *max_body as usize >= bodies.len()
    {
        return Err(TextParseError::new(
            *line,
            format!("body b{max_body} is sparse: body ids must be contiguous from b0"),
        ));
    }

    let mut circuit = CoordCircuit::new();
    for _ in 1..bodies.len() {
        circuit.add_body(CircuitBody::new());
    }
    // `MeasRegistry::reserve` panics when one id is bound to two different
    // qubits, so reject that conflict here with the offending line instead of
    // aborting on malformed input. Implied/implied conflicts were rejected at
    // their op line while parsing the body.
    let mut seen = implied.clone();
    for (record, line) in &extra {
        match seen.insert(record.id, record.qubit) {
            Some(qubit) if qubit != record.qubit => {
                return Err(TextParseError::new(
                    *line,
                    format!(
                        "measurement m{} is bound to two different qubits: \
                         ({},{}) and ({},{})",
                        record.id, qubit.x, qubit.y, record.qubit.x, record.qubit.y
                    ),
                ));
            }
            _ => {}
        }
    }

    let mut records: Vec<MeasRecord> = implied
        .into_iter()
        .map(|(id, qubit)| MeasRecord { id, qubit })
        .collect();
    records.extend(extra.into_iter().map(|(record, _)| record));
    for (body_id, (ops, _)) in bodies {
        *circuit
            .body_mut(BodyId(body_id))
            .expect("dense body ids were allocated above")
            .ops_mut() = ops;
    }
    circuit.register_measurement_records(&mut records);
    circuit
        .set_entry_body(entry)
        .expect("dense parsed bodies contain the entry");
    Ok(circuit)
}

fn parse_op(stmt: &Stmt<'_>) -> Result<Op> {
    let token = stmt.next_token("circuit op")?;
    let (head, probability) = parse_instruction_head(stmt, token)?;
    let op = match head {
        "TICK" if probability.is_none() => Op::Tick,
        "REPEAT" => {
            reject_probability_argument(stmt, token, probability)?;
            let repetitions = parse_int(stmt, stmt.next_token("repetition count")?)?;
            let body = BodyId(parse_prefixed(stmt, stmt.next_token("body id")?, 'b')?);
            Op::Repeat { body, repetitions }
        }
        "CPAULI" => {
            reject_probability_argument(stmt, token, probability)?;
            let mut corrections = Vec::new();
            while let Some(token) = stmt.peek() {
                stmt.advance();
                corrections.push(parse_correction(stmt, token)?);
            }
            Op::ConditionalPauli(corrections)
        }
        "MPP" => {
            reject_probability_argument(stmt, token, probability)?;
            let mut products = Vec::new();
            let mut measurements = Vec::new();
            while let Some(token) = stmt.peek() {
                stmt.advance();
                let (product, measurement) = token.rsplit_once(":").ok_or_else(|| {
                    stmt.err(format!("MPP product `{token}` is missing its `:mN` record"))
                })?;
                let product = parse_pauli_map(stmt, product)?;
                if product.is_empty() {
                    return Err(stmt.err("MPP product must contain a non-identity Pauli target"));
                }
                products.push(product);
                measurements.push(parse_measurement_output(stmt, measurement)?);
            }
            if products.is_empty() {
                return Err(stmt.err("MPP requires at least one product"));
            }
            Op::MPP {
                products,
                measurements,
            }
        }
        "M" | "MX" | "MY" => {
            let basis = match head {
                "MX" => PauliBasis::X,
                "MY" => PauliBasis::Y,
                _ => PauliBasis::Z,
            };
            let mut qubits = Vec::new();
            let mut measurements = Vec::new();
            while let Some(token) = stmt.peek() {
                stmt.advance();
                let (qubit, measurement) = token.split_once(":").ok_or_else(|| {
                    stmt.err(format!(
                        "measure target `{token}` is missing its `:mN` record"
                    ))
                })?;
                qubits.push(parse_coord2(stmt, qubit)?);
                measurements.push(parse_measurement_output(stmt, measurement)?);
            }
            Op::Measure {
                basis,
                qubits,
                measurements,
                flip_probability: probability.unwrap_or(0.0),
            }
        }
        "DEPOLARIZE1" => Op::Depolarize1 {
            probability: require_probability_argument(stmt, token, probability)?,
            qubits: parse_qubit_targets(stmt)?,
        },
        "DEPOLARIZE2" => {
            let qubits = parse_qubit_targets(stmt)?;
            if qubits.len() % 2 != 0 {
                return Err(stmt.err(format!(
                    "DEPOLARIZE2 requires an even target count, got {}",
                    qubits.len()
                )));
            }
            Op::Depolarize2 {
                probability: require_probability_argument(stmt, token, probability)?,
                qubits,
            }
        }
        "X_ERROR" | "Y_ERROR" | "Z_ERROR" => Op::PauliError {
            probability: require_probability_argument(stmt, token, probability)?,
            pauli: match head {
                "X_ERROR" => PauliBasis::X,
                "Y_ERROR" => PauliBasis::Y,
                _ => PauliBasis::Z,
            },
            qubits: parse_qubit_targets(stmt)?,
        },
        gate => {
            reject_probability_argument(stmt, token, probability)?;
            let gate =
                GateType::from_str(gate).map_err(|_| stmt.err(format!("unknown gate `{gate}`")))?;
            Op::Gate {
                gate,
                qubits: parse_qubit_targets(stmt)?,
            }
        }
    };
    stmt.finish()?;
    Ok(op)
}

fn parse_instruction_head<'a>(stmt: &Stmt<'_>, token: &'a str) -> Result<(&'a str, Option<f64>)> {
    let Some((name, argument)) = token.split_once('(') else {
        return Ok((token, None));
    };
    let argument = argument
        .strip_suffix(')')
        .ok_or_else(|| stmt.err(format!("malformed instruction argument in `{token}`")))?;
    if name.is_empty() || argument.is_empty() || argument.contains(['(', ')']) {
        return Err(stmt.err(format!("malformed instruction argument in `{token}`")));
    }
    let probability = argument
        .parse::<f64>()
        .map_err(|_| stmt.err(format!("invalid probability `{argument}`")))?;
    if !(0.0..=1.0).contains(&probability) {
        return Err(stmt.err(format!(
            "probability must be finite and in [0, 1], got `{argument}`"
        )));
    }
    Ok((name, Some(probability)))
}

fn reject_probability_argument(
    stmt: &Stmt<'_>,
    token: &str,
    probability: Option<f64>,
) -> Result<()> {
    if probability.is_some() {
        return Err(stmt.err(format!("instruction `{token}` does not take a probability")));
    }
    Ok(())
}

fn require_probability_argument(
    stmt: &Stmt<'_>,
    token: &str,
    probability: Option<f64>,
) -> Result<f64> {
    probability.ok_or_else(|| stmt.err(format!("instruction `{token}` requires a probability")))
}

fn parse_qubit_targets(stmt: &Stmt<'_>) -> Result<Vec<IVec2>> {
    let mut qubits = Vec::new();
    while let Some(token) = stmt.peek() {
        stmt.advance();
        qubits.push(parse_coord2(stmt, token)?);
    }
    Ok(qubits)
}

/// One `CPAULI` correction: `P[control]@(x,y)` with control `mN` or `vN`.
fn parse_correction(stmt: &Stmt<'_>, token: &str) -> Result<ConditionalCorrection> {
    let error = || {
        stmt.err(format!(
            "malformed correction `{token}`, expected `P[c]@(x,y)`"
        ))
    };
    let (pauli, rest) = token.split_once('[').ok_or_else(error)?;
    let (control, target) = rest.split_once("]@").ok_or_else(error)?;
    let control = parse_prefixed(stmt, control, 'm')?;
    Ok(ConditionalCorrection {
        pauli: parse_pauli_basis(stmt, pauli)?,
        control,
        target: parse_coord2(stmt, target)?,
    })
}

fn parse_flow(stmt: &Stmt<'_>) -> Result<Flow> {
    let start = parse_pauli_map(stmt, stmt.next_token("flow start")?)?;
    stmt.expect("->")?;
    let end = parse_pauli_map(stmt, stmt.next_token("flow end")?)?;
    let mut flow = Flow::new(start, end);
    while let Some(token) = stmt.peek() {
        stmt.advance();
        match token {
            "meas" => {
                let list = stmt.next_token("flow measurements")?;
                let mut measurements = Vec::new();
                for entry in list.split('*') {
                    measurements.push(parse_prefixed(stmt, entry, 'm')?);
                }
                flow = flow.with_measurements(measurements);
            }
            "center" => {
                let center = parse_coord2(stmt, stmt.next_token("flow center")?)?;
                flow = flow.with_center(center);
            }
            "sign" => {
                stmt.expect("-1")?;
                flow = flow.with_sign(true);
            }
            "discard" => flow = flow.with_marker(FlowMarker::Discard),
            "restart" => flow = flow.with_marker(FlowMarker::Restart),
            other => return Err(stmt.err(format!("unknown flow attribute `{other}`"))),
        }
    }
    Ok(flow)
}

// ==============================================================================
// Graph levels
// ==============================================================================

impl<'a> Parser<'a> {
    /// Parse the statements of one graph level up to its closing `}` and
    /// materialize the level with exact node ids: ids missing below the last
    /// live one become vacant slots (placeholder nodes added, then removed).
    fn parse_level(&mut self) -> Result<SubGraph> {
        let mut nodes: Vec<(u32, BloqNode)> = Vec::new();
        // Line numbers parallel to `nodes`/`edges`, to span rebuild errors.
        let mut node_lines: Vec<(u32, usize)> = Vec::new();
        let mut edges: Vec<(u32, u32, BloqEdge)> = Vec::new();
        let mut edge_lines: Vec<usize> = Vec::new();
        let mut value_output = None;
        let mut boundary_outputs = None;

        loop {
            let stmt = self.expect_stmt("a node, an edge, or `}`")?;
            if stmt.is_close() {
                break;
            }
            let head = stmt.next_token("node or edge")?;
            if head == "result" {
                if value_output.is_some() {
                    return Err(stmt.err("duplicate result declaration"));
                }
                let output = stmt.next_token("result node or false")?;
                value_output = Some(if output == "false" {
                    None
                } else {
                    Some(parse_value_ref(&stmt, output)?)
                });
                stmt.finish()?;
                continue;
            }
            if head == "bindings" {
                if boundary_outputs.is_some() {
                    return Err(stmt.err("duplicate bindings declaration"));
                }
                let mut outputs = Vec::new();
                while let Some(output) = stmt.peek() {
                    outputs.push(BloqNodeId(parse_prefixed(&stmt, output, 'n')?));
                    stmt.advance();
                }
                boundary_outputs = Some(outputs);
                continue;
            }
            let id = parse_prefixed(&stmt, head, 'n')?;
            if stmt.peek() == Some("->") {
                stmt.advance();
                let target = parse_prefixed(&stmt, stmt.next_token("edge target")?, 'n')?;
                edge_lines.push(stmt.line);
                edges.push((id, target, parse_edge(&stmt)?));
            } else {
                node_lines.push((id, stmt.line));
                let node = self.parse_node(&stmt)?;
                nodes.push((id, node));
            }
        }

        // The shared rebuild owns validation (duplicate ids, edge endpoints,
        // the vacancy cap); map its errors back to source lines here.
        let mut level = crate::ir::rebuild_level(nodes, edges).map_err(|error| {
            use crate::ir::LevelRebuildError;
            let line = match error {
                // The second occurrence is the offending statement.
                LevelRebuildError::DuplicateNode(id) => node_lines
                    .iter()
                    .filter(|(node, _)| *node == id)
                    .map(|&(_, line)| line)
                    .nth(1),
                LevelRebuildError::MissingEndpoint { index, .. } => edge_lines.get(index).copied(),
                // The node with the largest id is what stretched the id space.
                LevelRebuildError::ExcessiveVacancy { .. } => node_lines
                    .iter()
                    .max_by_key(|(node, _)| *node)
                    .map(|&(_, line)| line),
            };
            let line = line.unwrap_or(self.lines.len());
            TextParseError::new(line, error)
        })?;
        level.set_value_output(value_output.flatten());
        level.set_boundary_outputs(boundary_outputs.unwrap_or_default());
        Ok(level)
    }

    fn parse_node(&mut self, stmt: &Stmt<'_>) -> Result<BloqNode> {
        let kind = stmt.next_token("node kind")?;
        match kind {
            "quantum" => {
                stmt.expect("{")?;
                stmt.finish()?;
                self.parse_quantum_node()
            }
            "rus" => self.parse_region_node(stmt),
            _ => parse_classical_node(stmt, kind),
        }
    }

    fn parse_quantum_node(&mut self) -> Result<BloqNode> {
        let mut quantum = QuantumNode::default();
        let mut provenance = NodeProvenance::None;
        loop {
            let stmt = self.expect_stmt("a quantum-node statement or `}`")?;
            if stmt.is_close() {
                break;
            }
            let keyword = stmt.next_token("quantum-node statement")?;
            match keyword {
                "instance" => {
                    let id = parse_prefixed(&stmt, stmt.next_token("instance id")?, 'i')?;
                    let template = parse_prefixed(&stmt, stmt.next_token("template id")?, 't')?;
                    stmt.expect("@")?;
                    let offset = parse_coord2(&stmt, stmt.next_token("instance offset")?)?;
                    let provenance = if stmt.peek() == Some("from") {
                        stmt.advance();
                        match stmt.next_token("instance source")? {
                            "block" => crate::InstanceProvenance::Block {
                                source: parse_coord3(&stmt, stmt.next_token("source block")?)?,
                            },
                            "ideal-port" => crate::InstanceProvenance::IdealPortAdapter {
                                role: parse_port_role(&stmt)?,
                                source: parse_coord3(
                                    &stmt,
                                    stmt.next_token("source temporal Port")?,
                                )?,
                            },
                            "pipe" => crate::InstanceProvenance::Pipe {
                                src: parse_coord3(&stmt, stmt.next_token("pipe source")?)?,
                                dst: parse_coord3(&stmt, stmt.next_token("pipe destination")?)?,
                            },
                            "spatial-port" => {
                                let role = parse_port_role(&stmt)?;
                                let part = match stmt
                                    .next_token("spatial Port substitution part")?
                                {
                                    "cube" => crate::SpatialPortPart::Cube,
                                    "temporal-port" => crate::SpatialPortPart::TemporalPort,
                                    other => {
                                        return Err(stmt
                                            .err(format!("unknown spatial Port part `{other}`")));
                                    }
                                };
                                let source =
                                    parse_coord3(&stmt, stmt.next_token("source spatial Port")?)?;
                                crate::InstanceProvenance::SpatialPortSubstitution {
                                    source,
                                    role,
                                    part,
                                }
                            }
                            other => {
                                return Err(stmt.err(format!("unknown instance source `{other}`")));
                            }
                        }
                    } else {
                        crate::InstanceProvenance::Source
                    };
                    stmt.finish()?;
                    quantum.instances.push(TemplateInstance {
                        id: TemplateInstanceId(id),
                        template_id: TemplateId(template),
                        offset,
                        provenance,
                    });
                }
                "detector" => {
                    let parity = parse_parity(
                        &stmt,
                        stmt.next_token("detector parity")?,
                        parse_instance_measurement,
                    )?;
                    let coords = parse_opt_detector_coords(&stmt)?;
                    stmt.finish()?;
                    quantum.detectors.push(NodeDetector { parity, coords });
                }
                "use" => {
                    let bundle = DetectorBundleId(parse_prefixed(
                        &stmt,
                        stmt.next_token("bundle id")?,
                        'b',
                    )?);
                    let mut instances = Vec::new();
                    while stmt.peek() != Some("@") {
                        instances.push(TemplateInstanceId(parse_prefixed(
                            &stmt,
                            stmt.next_token("owner instance or `@`")?,
                            'i',
                        )?));
                    }
                    stmt.expect("@")?;
                    let offset = parse_coord2(&stmt, stmt.next_token("bundle offset")?)?;
                    stmt.finish()?;
                    quantum.detector_bundles.push(DetectorBundleUse {
                        bundle,
                        instances,
                        offset,
                    });
                }
                "restart" => {
                    let parity = parse_parity(
                        &stmt,
                        stmt.next_token("restart parity")?,
                        parse_instance_measurement,
                    )?;
                    stmt.finish()?;
                    quantum.restarts.push(NodeRestart { parity });
                }
                "timeline" => {
                    stmt.expect("rounds")?;
                    let mut layer_round_ends = Vec::new();
                    while let Some(end) = stmt.peek() {
                        stmt.advance();
                        layer_round_ends.push(parse_int(&stmt, end)?);
                    }
                    if quantum
                        .timeline
                        .replace(QuantumTimeline { layer_round_ends })
                        .is_some()
                    {
                        return Err(stmt.err("duplicate quantum timeline"));
                    }
                    stmt.finish()?;
                }
                "guard" => {
                    let input = parse_int(&stmt, stmt.next_token("membership input slot")?)?;
                    let mut guard = crate::QuantumGuard {
                        input,
                        instances: Vec::new(),
                        ..Default::default()
                    };
                    while let Some(token) = stmt.peek() {
                        stmt.advance();
                        match token.as_bytes().first() {
                            Some(b'i') => guard
                                .instances
                                .push(TemplateInstanceId(parse_prefixed(&stmt, token, 'i')?)),
                            Some(b'd') => guard.detectors.push(parse_prefixed(&stmt, token, 'd')?),
                            Some(b'b') => guard
                                .detector_bundles
                                .push(parse_prefixed(&stmt, token, 'b')?),
                            Some(b'r') => guard.restarts.push(parse_prefixed(&stmt, token, 'r')?),
                            Some(b'x') => {
                                let (target, parity) = token[1..]
                                    .split_once('=')
                                    .ok_or_else(|| stmt.err("expected xdN=parity or xrN=parity"))?;
                                let parity =
                                    parse_parity(&stmt, parity, parse_instance_measurement)?;
                                if target.starts_with('d') {
                                    guard
                                        .detector_parities
                                        .push((parse_prefixed(&stmt, target, 'd')?, parity));
                                } else {
                                    guard
                                        .restart_parities
                                        .push((parse_prefixed(&stmt, target, 'r')?, parity));
                                }
                            }
                            _ => {
                                return Err(stmt.err(
                                    "expected guarded instance iN, detector dN, bundle use bN, or restart rN",
                                ));
                            }
                        }
                    }
                    stmt.finish()?;
                    quantum.guards.push(guard);
                }
                "from" => {
                    provenance = parse_provenance(&stmt)?;
                    stmt.finish()?;
                }
                other => {
                    return Err(stmt.err(format!("unknown quantum-node statement `{other}`")));
                }
            }
        }
        Ok(BloqNode::quantum(quantum).with_provenance(provenance))
    }

    fn parse_region_node(&mut self, stmt: &Stmt<'_>) -> Result<BloqNode> {
        // Header: `<expr> [source nK] {` — the expression runs until a
        // terminator keyword. `source` binds body-local restart inputs.
        let expr = parse_expr_until(stmt, &["source", "when", "{"])?;
        let restart_source = if stmt.peek() == Some("source") {
            stmt.advance();
            Some(parse_value_ref(
                stmt,
                stmt.next_token("restart source node")?,
            )?)
        } else {
            None
        };
        let activation = if stmt.peek() == Some("when") {
            stmt.advance();
            Some(parse_prefixed(
                stmt,
                stmt.next_token("activation input")?,
                'v',
            )?)
        } else {
            None
        };
        stmt.expect("{")?;
        stmt.finish()?;

        let mut provenance = NodeProvenance::None;
        let mut next = self.expect_stmt("a region body or `}`")?;
        if next.peek() == Some("from") {
            next.advance();
            provenance = parse_provenance(&next)?;
            next.finish()?;
            next = self.expect_stmt("a region body or `}`")?;
        }

        next.expect("body")?;
        next.expect("{")?;
        next.finish()?;
        let region = RegionNode::RepeatUntilSuccess {
            body: self.parse_level()?,
            restart_condition: expr,
            restart_source,
        };

        let close = self.expect_stmt("`}` closing the region node")?;
        if !close.is_close() {
            return Err(close.err("expected `}` closing the region node"));
        }
        Ok(BloqNode {
            activation,
            kind: BloqNodeKind::Region(region),
            provenance,
        })
    }
}

fn parse_classical_node(stmt: &Stmt<'_>, kind: &str) -> Result<BloqNode> {
    let classical = match kind {
        "compute" => ClassicalNode::Compute {
            expr: parse_expr_until(stmt, &["when", "from"])?,
        },
        "observable" => {
            let token = stmt.next_token("observable index or fragment")?;
            let index = if token == "fragment" {
                None
            } else {
                Some(parse_int(stmt, token)?)
            };
            let measurements = if stmt.peek() == Some("measurements") {
                stmt.advance();
                parse_classical_measurements(stmt)?
            } else {
                Vec::new()
            };
            let operators = if stmt.peek() == Some("operators") {
                stmt.advance();
                parse_classical_operators(stmt)?
            } else {
                Vec::new()
            };
            ClassicalNode::Observable {
                index,
                measurements,
                operators,
            }
        }
        "discard" => ClassicalNode::Discard {
            condition: parse_expr_until(stmt, &["when", "from"])?,
        },
        other => return Err(stmt.err(format!("unknown node kind `{other}`"))),
    };

    let activation = if stmt.peek() == Some("when") {
        stmt.advance();
        Some(parse_prefixed(stmt, stmt.next_token("guard input")?, 'v')?)
    } else {
        None
    };
    let provenance = if stmt.peek() == Some("from") {
        stmt.advance();
        parse_provenance(stmt)?
    } else {
        NodeProvenance::None
    };
    stmt.finish()?;
    Ok(BloqNode {
        activation,
        kind: BloqNodeKind::Classical(classical.into()),
        provenance,
    })
}

fn parse_classical_measurements(stmt: &Stmt<'_>) -> Result<Vec<InstanceMeasurement>> {
    let Some(token) = stmt.peek() else {
        return Ok(Vec::new());
    };
    if matches!(token, "operators" | "from" | "when") {
        return Ok(Vec::new());
    }
    stmt.advance();
    token
        .split('*')
        .map(|entry| parse_instance_measurement(stmt, entry))
        .collect()
}

fn parse_classical_operators(stmt: &Stmt<'_>) -> Result<Vec<InstanceBoundaryOperator>> {
    // A trailing comma on a Pauli-map token means another operator follows.
    let mut operators = Vec::new();
    while !matches!(stmt.peek(), None | Some("from" | "when")) {
        let instance =
            TemplateInstanceId(parse_prefixed(stmt, stmt.next_token("instance id")?, 'i')?);
        let face = match stmt.next_token("`input` or `output`")? {
            "input" => BoundaryFace::Input,
            "output" => BoundaryFace::Output,
            other => return Err(stmt.err(format!("unknown boundary face `{other}`"))),
        };
        let token = stmt.next_token("boundary operator")?;
        let (map_token, more) = token
            .strip_suffix(',')
            .map_or((token, false), |rest| (rest, true));
        operators.push(InstanceBoundaryOperator {
            instance,
            face,
            operator: parse_pauli_map(stmt, map_token)?,
        });
        if !more {
            break;
        }
    }
    Ok(operators)
}

fn parse_edge(stmt: &Stmt<'_>) -> Result<BloqEdge> {
    let edge = match stmt.next_token("edge kind")? {
        "order" => BloqEdge::Order,
        kind @ ("value" | "compose") => {
            let slot = parse_int(stmt, stmt.next_token("input slot")?)?;
            let output = if kind == "value" && matches!(stmt.peek(), Some("corrected" | "flip")) {
                parse_output(stmt, stmt.next_token("output port")?)?
            } else {
                ObservableOutput::Corrected
            };
            let role = match stmt.peek() {
                Some("feedback") => {
                    stmt.advance();
                    ValueRole::FeedbackFold {
                        action: parse_int(stmt, stmt.next_token("feedback action")?)?,
                    }
                }
                Some("readout") => {
                    stmt.advance();
                    ValueRole::ReadoutFold
                }
                _ => ValueRole::Data,
            };
            if kind == "compose" {
                BloqEdge::Compose { slot, role }
            } else {
                BloqEdge::Value { slot, role, output }
            }
        }
        "quantum" => {
            let mut pipes = Vec::new();
            while let Some(token) = stmt.peek() {
                if token == "when" {
                    break;
                }
                stmt.advance();
                let pipe = parse_pipe(stmt, token)?;
                // Each pipe's padding, when recorded, immediately follows it.
                let padding = if stmt.peek() == Some("padding") {
                    stmt.advance();
                    stmt.expect("offset")?;
                    let offset = parse_coord2(stmt, stmt.next_token("padding offset")?)?;
                    stmt.expect("one")?;
                    let one_round = parse_prefixed(stmt, stmt.next_token("template id")?, 't')?;
                    stmt.expect("loop")?;
                    let looped = parse_prefixed(stmt, stmt.next_token("template id")?, 't')?;
                    Some(PipePadding {
                        offset,
                        one_round: TemplateId(one_round),
                        looped: TemplateId(looped),
                    })
                } else {
                    None
                };
                pipes.push(PipeSeam { pipe, padding });
            }
            let guard = if stmt.peek() == Some("when") {
                stmt.advance();
                Some(parse_value_ref(
                    stmt,
                    stmt.next_token("seam guard producer")?,
                )?)
            } else {
                None
            };
            BloqEdge::Quantum(Box::new(QuantumEdge { pipes, guard }))
        }
        other => return Err(stmt.err(format!("unknown edge kind `{other}`"))),
    };
    stmt.finish()?;
    Ok(edge)
}

fn parse_output(stmt: &Stmt<'_>, token: &str) -> Result<ObservableOutput> {
    match token {
        "corrected" => Ok(ObservableOutput::Corrected),
        "flip" => Ok(ObservableOutput::Flip),
        _ => Err(stmt.err(format!("unknown output port `{token}`"))),
    }
}

fn parse_value_ref(stmt: &Stmt<'_>, token: &str) -> Result<ValueRef> {
    let (node, output) = match token.split_once(':') {
        Some((node, output)) => (node, parse_output(stmt, output)?),
        None => (token, ObservableOutput::Corrected),
    };
    Ok(ValueRef {
        node: BloqNodeId(parse_prefixed(stmt, node, 'n')?),
        output,
    })
}

// ==============================================================================
// Provenance
// ==============================================================================

fn parse_provenance(stmt: &Stmt<'_>) -> Result<NodeProvenance> {
    let provenance = match stmt.next_token("provenance kind")? {
        "blocks" => {
            let mut members = Vec::new();
            while let Some(token) = stmt.peek() {
                stmt.advance();
                members.push(SourceBlockRef {
                    pos: parse_coord3(stmt, token)?,
                });
            }
            NodeProvenance::BlockComponent { members }
        }
        "pipe" => NodeProvenance::TemporalPipe {
            pipe: parse_pipe(stmt, stmt.next_token("pipe")?)?,
        },
        "spatial-port" => {
            let role = parse_port_role(stmt)?;
            let source = parse_coord3(stmt, stmt.next_token("source spatial Port")?)?;
            NodeProvenance::SpatialPortSubstitution { source, role }
        }
        "padding" => {
            let pipe = parse_pipe(stmt, stmt.next_token("pipe")?)?;
            stmt.expect("rounds")?;
            let rounds = parse_int(stmt, stmt.next_token("round count")?)?;
            NodeProvenance::MemoryPadding { pipe, rounds }
        }
        "generator" => NodeProvenance::Generator {
            ordinal: parse_int(stmt, stmt.next_token("generator ordinal")?)?,
        },
        "action" => NodeProvenance::Action {
            ordinal: parse_int(stmt, stmt.next_token("action ordinal")?)?,
        },
        "selector" => NodeProvenance::BranchSelector {
            name: decode_metadata_token(stmt.next_token("branch name")?)
                .map_err(|message| stmt.err(message))?,
        },
        "frame" => {
            let basis = match stmt.next_token("frame basis")? {
                "x" => Basis::X,
                "z" => Basis::Z,
                other => return Err(stmt.err(format!("unknown frame basis `{other}`"))),
            };
            let port = parse_coord3(stmt, stmt.next_token("frame output port")?)?;
            NodeProvenance::OutputFrame { port, basis }
        }
        other => return Err(stmt.err(format!("unknown provenance kind `{other}`"))),
    };
    Ok(provenance)
}

// ==============================================================================
// Token-level parsers
// ==============================================================================

/// `prefix(args)` → `Some(args)`, e.g. `call_args("body(b1)", "body")` →
/// `Some("b1")`.
fn call_args<'a>(token: &'a str, name: &str) -> Option<&'a str> {
    token
        .strip_prefix(name)?
        .strip_prefix('(')?
        .strip_suffix(')')
}

/// Split comma-separated call arguments into exactly `N` pieces.
fn parse_call_args<'a, const N: usize>(stmt: &Stmt<'_>, args: &'a str) -> Result<[&'a str; N]> {
    let pieces: Vec<&str> = args.split(',').collect();
    <[&str; N]>::try_from(pieces).map_err(|pieces: Vec<&str>| {
        stmt.err(format!("expected {N} argument(s), found {}", pieces.len()))
    })
}

fn parse_int<T: FromStr>(stmt: &Stmt<'_>, token: &str) -> Result<T> {
    token
        .parse()
        .map_err(|_| stmt.err(format!("invalid number `{token}`")))
}

/// `x123` → `123` for the id sigils (`n`, `t`, `i`, `m`, `b`, `s`). Digits
/// only — `u32::from_str` would also accept a leading `+`, which the writer
/// never emits.
fn parse_prefixed(stmt: &Stmt<'_>, token: &str, prefix: char) -> Result<u32> {
    token
        .strip_prefix(prefix)
        .filter(|digits| !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit()))
        .and_then(|digits| digits.parse().ok())
        .ok_or_else(|| stmt.err(format!("expected `{prefix}<N>`, found `{token}`")))
}

/// The next token as a spatial-Port role, shared by the node-level and
/// instance-level `spatial-port` provenance forms so both report it alike.
fn parse_port_role(stmt: &Stmt<'_>) -> Result<crate::PortRole> {
    stmt.next_token("spatial Port role")?
        .parse()
        .map_err(|_| stmt.err("invalid spatial Port role"))
}

fn parse_measurement_output(stmt: &Stmt<'_>, token: &str) -> Result<u32> {
    let measurement = parse_prefixed(stmt, token, 'm')?;
    if measurement == u32::MAX {
        return Err(stmt.err(format!(
            "measurement id m{measurement} exhausts the measurement id space"
        )));
    }
    Ok(measurement)
}

fn parse_coord2(stmt: &Stmt<'_>, token: &str) -> Result<IVec2> {
    let [x, y] = parse_int_tuple(stmt, token)?;
    Ok(IVec2::new(x, y))
}

fn parse_coord3(stmt: &Stmt<'_>, token: &str) -> Result<IVec3> {
    let [x, y, z] = parse_int_tuple(stmt, token)?;
    Ok(IVec3::new(x, y, z))
}

fn parse_int_tuple<const N: usize>(stmt: &Stmt<'_>, token: &str) -> Result<[i32; N]> {
    let inner = token
        .strip_prefix('(')
        .and_then(|rest| rest.strip_suffix(')'))
        .ok_or_else(|| stmt.err(format!("expected a coordinate, found `{token}`")))?;
    let mut out = [0i32; N];
    let mut pieces = inner.split(',');
    for slot in &mut out {
        let piece = pieces
            .next()
            .ok_or_else(|| stmt.err(format!("coordinate `{token}` has too few components")))?;
        *slot = parse_int(stmt, piece)?;
    }
    if pieces.next().is_some() {
        return Err(stmt.err(format!("coordinate `{token}` has too many components")));
    }
    Ok(out)
}

/// `(x,y,z)>(x,y,z)` with an optional trailing `h` for a hadamard pipe.
fn parse_pipe(stmt: &Stmt<'_>, token: &str) -> Result<TemporalPipeRef> {
    let (token, hadamard) = token
        .strip_suffix('h')
        .map_or((token, false), |rest| (rest, true));
    let (src, dst) = token
        .split_once('>')
        .ok_or_else(|| stmt.err(format!("expected a pipe `(..)>(..)`, found `{token}`")))?;
    Ok(TemporalPipeRef {
        src: parse_coord3(stmt, src)?,
        dst: parse_coord3(stmt, dst)?,
        hadamard,
    })
}

fn parse_pauli(stmt: &Stmt<'_>, token: &str) -> Result<Pauli> {
    Pauli::from_str(token).map_err(|_| stmt.err(format!("invalid Pauli `{token}`")))
}

fn parse_pauli_basis(stmt: &Stmt<'_>, token: &str) -> Result<PauliBasis> {
    PauliBasis::from_str(token).map_err(|_| stmt.err(format!("invalid Pauli basis `{token}`")))
}

/// `X(0,0)*Z(1,2)` (or `_` for the empty map).
fn parse_pauli_map(stmt: &Stmt<'_>, token: &str) -> Result<PauliMap> {
    if token == "_" {
        return Ok(PauliMap::empty());
    }
    let mut map = PauliMap::empty();
    for entry in token.split('*') {
        let (pauli, coord) = entry
            .split_at_checked(1)
            .ok_or_else(|| stmt.err(format!("malformed Pauli map entry `{entry}`")))?;
        map.insert(parse_coord2(stmt, coord)?, parse_pauli(stmt, pauli)?);
    }
    Ok(map)
}

fn parse_meas_term(stmt: &Stmt<'_>, token: &str) -> Result<u32> {
    parse_prefixed(stmt, token, 'm')
}

/// `iN:mK`.
fn parse_instance_measurement(stmt: &Stmt<'_>, token: &str) -> Result<InstanceMeasurement> {
    let (instance, measurement) = token
        .split_once(':')
        .ok_or_else(|| stmt.err(format!("expected `iN:mK`, found `{token}`")))?;
    Ok(InstanceMeasurement {
        instance: TemplateInstanceId(parse_prefixed(stmt, instance, 'i')?),
        measurement: parse_prefixed(stmt, measurement, 'm')?,
    })
}

/// `oN:mK`, where `oN` is a bundle-local owner slot.
fn parse_bundle_measurement(stmt: &Stmt<'_>, token: &str) -> Result<BundleMeasurement> {
    let (owner, measurement) = token
        .split_once(':')
        .ok_or_else(|| stmt.err(format!("expected `oN:mK`, found `{token}`")))?;
    Ok(BundleMeasurement {
        owner: parse_prefixed(stmt, owner, 'o')?,
        measurement: parse_prefixed(stmt, measurement, 'm')?,
    })
}

/// Optional `-` sign followed by `term*term*…`, with terms `sN` (loop state)
/// or a measurement (via `measurement`); `0` is the empty parity.
fn parse_parity<M: Copy + Ord>(
    stmt: &Stmt<'_>,
    token: &str,
    measurement: impl Fn(&Stmt<'_>, &str) -> Result<M>,
) -> Result<DetectorParity<M>> {
    let (sign, token) = token
        .strip_prefix('-')
        .map_or((false, token), |token| (true, token));
    if token == "0" {
        return Ok(DetectorParity::default().with_sign(sign));
    }
    let mut terms = Vec::new();
    for entry in token.split('*') {
        // A loop-state term is `s<N>`; instance measurements contain `:` and
        // template measurements start with `m`, so the sigil disambiguates.
        if entry.starts_with('s') {
            terms.push(DetectorTerm::LoopState(LoopStateId(parse_prefixed(
                stmt, entry, 's',
            )?)));
        } else {
            terms.push(DetectorTerm::Measurement(measurement(stmt, entry)?));
        }
    }
    Ok(DetectorParity::from_terms(terms).with_sign(sign))
}

/// Optional ` @ (c0,c1,…)` detector coordinates.
fn parse_opt_detector_coords(stmt: &Stmt<'_>) -> Result<Option<bloq_circuit::DetectorCoords>> {
    if stmt.peek() != Some("@") {
        return Ok(None);
    }
    stmt.advance();
    let token = stmt.next_token("detector coordinates")?;
    let inner = token
        .strip_prefix('(')
        .and_then(|rest| rest.strip_suffix(')'))
        .ok_or_else(|| stmt.err(format!("expected coordinates, found `{token}`")))?;
    let mut coords = bloq_circuit::DetectorCoords::new();
    if !inner.is_empty() {
        for piece in inner.split(',') {
            coords.push(
                piece
                    .parse()
                    .map_err(|_| stmt.err(format!("invalid coordinate `{piece}`")))?,
            );
        }
    }
    Ok(Some(coords))
}

// ==============================================================================
// Classical expressions
// ==============================================================================

/// Parse an expression from the statement's remaining tokens, stopping (without
/// consuming) at any of the `terminators` or at end of line.
fn parse_expr_until(stmt: &Stmt<'_>, terminators: &[&str]) -> Result<ClassicalExpr> {
    let mut source = String::new();
    while let Some(token) = stmt.peek() {
        if terminators.contains(&token) {
            break;
        }
        stmt.advance();
        source.push_str(token);
        source.push(' ');
    }
    parse_expr(stmt, &source)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExprToken {
    In(u32),
    Const(bool),
    Not,
    Xor,
    And,
    Or,
    XorList,
    AndList,
    OrList,
    Select,
    Parity,
    Comma,
    Open,
    Close,
}

fn lex_expr(stmt: &Stmt<'_>, source: &str) -> Result<Vec<ExprToken>> {
    let mut tokens = Vec::new();
    let bytes = source.as_bytes();
    let mut index = 0;
    while index < bytes.len() {
        let rest = &source[index..];
        let (token, width) = match bytes[index] {
            b' ' => {
                index += 1;
                continue;
            }
            b'!' => (ExprToken::Not, 1),
            b'^' => (ExprToken::Xor, 1),
            b'&' => (ExprToken::And, 1),
            b'|' => (ExprToken::Or, 1),
            b',' => (ExprToken::Comma, 1),
            b'x' if rest.starts_with("xor") => (ExprToken::XorList, 3),
            b'a' if rest.starts_with("and") => (ExprToken::AndList, 3),
            b'o' if rest.starts_with("or") => (ExprToken::OrList, 2),
            b's' if rest.starts_with("select") => (ExprToken::Select, 6),
            b'p' if rest.starts_with("parity") => (ExprToken::Parity, 6),
            b'(' => (ExprToken::Open, 1),
            b')' => (ExprToken::Close, 1),
            b'0' => (ExprToken::Const(false), 1),
            b'1' => (ExprToken::Const(true), 1),
            _ => {
                let digits = rest
                    .strip_prefix("in")
                    .map(|after| after.bytes().take_while(u8::is_ascii_digit).count())
                    .filter(|&len| len > 0)
                    .ok_or_else(|| stmt.err(format!("invalid expression at `{rest}`")))?;
                let slot = rest[2..2 + digits]
                    .parse()
                    .map_err(|_| stmt.err(format!("invalid input slot in `{rest}`")))?;
                (ExprToken::In(slot), 2 + digits)
            }
        };
        tokens.push(token);
        index += width;
    }
    Ok(tokens)
}

/// Precedence-climbing over the lexed tokens; mirrors the writer's levels
/// (`|` < `^` < `&` < `!`).
struct ExprParser<'s, 'a> {
    stmt: &'s Stmt<'a>,
    tokens: Vec<ExprToken>,
    next: usize,
}

fn parse_expr(stmt: &Stmt<'_>, source: &str) -> Result<ClassicalExpr> {
    let mut parser = ExprParser {
        stmt,
        tokens: lex_expr(stmt, source)?,
        next: 0,
    };
    let expr = parser.parse_or()?;
    if parser.next != parser.tokens.len() {
        return Err(stmt.err("trailing tokens in expression"));
    }
    Ok(expr)
}

impl ExprParser<'_, '_> {
    fn peek(&self) -> Option<ExprToken> {
        self.tokens.get(self.next).copied()
    }

    fn parse_or(&mut self) -> Result<ClassicalExpr> {
        self.parse_binop(ExprToken::Or, ClassicalExpr::Or, Self::parse_xor)
    }

    fn parse_xor(&mut self) -> Result<ClassicalExpr> {
        self.parse_binop(ExprToken::Xor, ClassicalExpr::Xor, Self::parse_and)
    }

    fn parse_and(&mut self) -> Result<ClassicalExpr> {
        self.parse_binop(ExprToken::And, ClassicalExpr::And, Self::parse_not)
    }

    /// One flat precedence level: `next (token next)*`.
    fn parse_binop(
        &mut self,
        token: ExprToken,
        build: fn(Box<[ClassicalExpr]>) -> ClassicalExpr,
        next: fn(&mut Self) -> Result<ClassicalExpr>,
    ) -> Result<ClassicalExpr> {
        let first = next(self)?;
        if self.peek() != Some(token) {
            return Ok(first);
        }
        let mut operands = vec![first];
        while self.peek() == Some(token) {
            self.next += 1;
            operands.push(next(self)?);
        }
        Ok(build(operands.into_boxed_slice()))
    }

    fn parse_not(&mut self) -> Result<ClassicalExpr> {
        if self.peek() == Some(ExprToken::Not) {
            self.next += 1;
            return Ok(ClassicalExpr::Not(Box::new(self.parse_not()?)));
        }
        self.parse_atom()
    }

    fn parse_atom(&mut self) -> Result<ClassicalExpr> {
        let token = self
            .peek()
            .ok_or_else(|| self.stmt.err("expression ended unexpectedly"))?;
        self.next += 1;
        match token {
            ExprToken::In(slot) => Ok(ClassicalExpr::In(slot)),
            ExprToken::Const(bit) => Ok(ClassicalExpr::Const(bit)),
            ExprToken::XorList
            | ExprToken::AndList
            | ExprToken::OrList
            | ExprToken::Select
            | ExprToken::Parity => {
                if self.peek() != Some(ExprToken::Open) {
                    return Err(self.stmt.err("expected `(` after expression function"));
                }
                self.next += 1;
                let mut operands = Vec::new();
                if self.peek() != Some(ExprToken::Close) {
                    loop {
                        operands.push(self.parse_or()?);
                        if self.peek() != Some(ExprToken::Comma) {
                            break;
                        }
                        self.next += 1;
                    }
                }
                if self.peek() != Some(ExprToken::Close) {
                    return Err(self.stmt.err("unclosed expression function"));
                }
                self.next += 1;
                Ok(match token {
                    ExprToken::XorList => ClassicalExpr::Xor(operands.into_boxed_slice()),
                    ExprToken::AndList => ClassicalExpr::And(operands.into_boxed_slice()),
                    ExprToken::OrList => ClassicalExpr::Or(operands.into_boxed_slice()),
                    ExprToken::Parity => {
                        let mut operands = operands.into_iter();
                        let Some(ClassicalExpr::Const(constant)) = operands.next() else {
                            return Err(self
                                .stmt
                                .err("parity expects a constant followed by input slots"));
                        };
                        let inputs = operands
                            .map(|operand| match operand {
                                ClassicalExpr::In(slot) => Ok(slot),
                                _ => Err(self.stmt.err("parity operands must be input slots")),
                            })
                            .collect::<Result<Box<[_]>>>()?;
                        ClassicalExpr::Parity { inputs, constant }
                    }
                    _ => ClassicalExpr::Select(Box::new(operands.try_into().map_err(|_| {
                        self.stmt
                            .err("select expects condition, when_false, when_true")
                    })?)),
                })
            }
            ExprToken::Open => {
                let expr = self.parse_or()?;
                if self.peek() != Some(ExprToken::Close) {
                    return Err(self.stmt.err("unclosed `(` in expression"));
                }
                self.next += 1;
                Ok(expr)
            }
            _ => Err(self.stmt.err("expected an expression atom")),
        }
    }
}
