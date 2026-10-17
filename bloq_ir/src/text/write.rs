//! Writer half of the text exchange format. See the module doc in
//! [`super`] for the grammar; the parser in [`super::parse`] is the other
//! half, and the two must agree token for token.

use std::fmt::{self, Write as _};

use bloq_circuit::{
    CoordCircuit, DetectorParity, DetectorTerm, Flow, FlowMarker, Op, PauliBasis, PauliMap,
    measurement_gate_name,
};
use glam::{IVec2, IVec3};

use crate::{
    Basis, Bloq, BloqEdge, BloqEdgeRef, BloqNode, BloqNodeId, BloqTemplate, BundleMeasurement,
    ClassicalExpr, ClassicalNode, InstanceBoundaryOperator, InstanceMeasurement, MetadataValue,
    NodeProvenance, ObservableOutput, RegionNode, SubGraph, ValueRef, ValueRole,
};

use super::{BLOQ_TEXT_VERSION, encode_metadata_token, parse::implied_measurement_qubits};

pub(super) fn write_bloq_text(bloq: &Bloq) -> String {
    let mut out = String::new();
    write_program(&mut out, bloq).expect("writing to a String cannot fail");
    out
}

fn write_program(f: &mut String, bloq: &Bloq) -> fmt::Result {
    writeln!(f, "BLOQIR {BLOQ_TEXT_VERSION}")?;
    for (key, value) in bloq.metadata() {
        let key = encode_metadata_token(key);
        match value {
            MetadataValue::U64(value) => writeln!(f, "metadata {key} u64 {value}")?,
            MetadataValue::String(value) => {
                writeln!(f, "metadata {key} string {}", encode_metadata_token(value))?
            }
        }
    }
    for input in bloq.logical_inputs() {
        write!(
            f,
            "logical-input {} instance i{} x ",
            Coord3(input.port),
            input.instance.0,
        )?;
        write_pauli_map(f, &input.x)?;
        write!(f, " z ")?;
        write_pauli_map(f, &input.z)?;
        writeln!(f)?;
    }
    for output in bloq.logical_outputs() {
        write!(
            f,
            "logical-output {} instance i{} x ",
            Coord3(output.port),
            output.instance.0
        )?;
        write_pauli_map(f, &output.x)?;
        write!(f, " z ")?;
        write_pauli_map(f, &output.z)?;
        writeln!(f)?;
    }
    for (id, template) in bloq.templates().iter() {
        writeln!(f)?;
        writeln!(f, "template t{} {{", id.0)?;
        write_template(f, template)?;
        writeln!(f, "}}")?;
    }
    for (id, bundle) in bloq.detector_bundles().iter() {
        writeln!(f)?;
        write!(f, "bundle b{} owners", id.0)?;
        for template in bundle.owner_templates() {
            write!(f, " t{}", template.0)?;
        }
        writeln!(f, " {{")?;
        for detector in bundle.detectors() {
            write!(f, "  detector ")?;
            write_bundle_parity(f, &detector.parity)?;
            write_detector_coords(f, detector.coords.as_deref())?;
            writeln!(f)?;
        }
        writeln!(f, "}}")?;
    }

    writeln!(f)?;
    writeln!(f, "graph {{")?;
    write_level(f, bloq.top(), 1)?;
    writeln!(f, "}}")?;
    Ok(())
}

// ==============================================================================
// Templates
// ==============================================================================

fn write_template(f: &mut String, template: &BloqTemplate) -> fmt::Result {
    write_circuit(f, &template.circuit)?;

    for detector in &template.detectors {
        write!(f, "  detector")?;
        if let crate::TemplateDetectorScope::RepeatBody { body } = detector.scope {
            write!(f, " body(b{})", body.0)?;
        }
        write!(f, " ")?;
        write_template_parity(f, &detector.parity)?;
        write_detector_coords(f, detector.coords.as_deref())?;
        writeln!(f)?;
    }
    for state in &template.repeat_states {
        write!(f, "  loop(b{},s{}) init ", state.body.0, state.state.0)?;
        write_template_parity(f, &state.initial)?;
        write!(f, " next ")?;
        write_template_parity(f, &state.next)?;
        writeln!(f)?;
    }
    for flow in &template.boundary_flows {
        write!(f, "  ")?;
        write_flow(f, flow)?;
        writeln!(f)?;
    }
    for restart in &template.restarts {
        write!(f, "  restart ")?;
        write_template_parity(f, &restart.parity)?;
        writeln!(f)?;
    }
    Ok(())
}

fn write_circuit(f: &mut String, circuit: &CoordCircuit) -> fmt::Result {
    for body_index in 0..circuit.body_count() {
        let body_id = bloq_circuit::BodyId(body_index as u32);
        if body_id == circuit.entry_body() {
            if body_index == 0 {
                writeln!(f, "  circuit {{")?;
            } else {
                writeln!(f, "  circuit b{body_index} {{")?;
            }
        } else {
            writeln!(f, "  body b{body_index} {{")?;
        }
        let body = circuit.body(body_id).expect("body index is in bounds");
        for op in body.ops() {
            write!(f, "    ")?;
            write_op(f, op)?;
            writeln!(f)?;
        }
        writeln!(f, "  }}")?;
    }

    // Registry records the ops do not imply (reserved ids, exotic producers).
    let implied = implied_measurement_qubits(
        (0..circuit.body_count())
            .filter_map(|index| circuit.body(bloq_circuit::BodyId(index as u32)))
            .flat_map(bloq_circuit::CircuitBody::ops),
    );
    for record in circuit.meas_registry().records() {
        if implied.get(&record.id) != Some(&record.qubit) {
            writeln!(f, "  meas m{} @ {}", record.id, Coord2(record.qubit))?;
        }
    }
    Ok(())
}

fn write_op(f: &mut String, op: &Op) -> fmt::Result {
    match op {
        Op::Gate { gate, qubits } => {
            write!(f, "{gate}")?;
            for &qubit in qubits {
                write!(f, " {}", Coord2(qubit))?;
            }
        }
        Op::Measure {
            basis,
            qubits,
            measurements,
            flip_probability,
        } => {
            assert_eq!(
                qubits.len(),
                measurements.len(),
                "a Measure op allocates one measurement id per target qubit"
            );
            write!(f, "{}", measurement_gate_name(*basis))?;
            if *flip_probability != 0.0 {
                write!(f, "({flip_probability})")?;
            }
            for (&qubit, &measurement) in qubits.iter().zip(measurements) {
                write!(f, " {}:m{measurement}", Coord2(qubit))?;
            }
        }
        Op::MPP {
            products,
            measurements,
        } => {
            assert_eq!(
                products.len(),
                measurements.len(),
                "an MPP op allocates one measurement id per product"
            );
            write!(f, "MPP")?;
            for (product, &measurement) in products.iter().zip(measurements) {
                write!(f, " ")?;
                write_pauli_map(f, product)?;
                write!(f, ":m{measurement}")?;
            }
        }
        Op::Tick => write!(f, "TICK")?,
        Op::Repeat { body, repetitions } => write!(f, "REPEAT {repetitions} b{}", body.0)?,
        Op::ConditionalPauli(corrections) => {
            write!(f, "CPAULI")?;
            for correction in corrections {
                write!(
                    f,
                    " {}[m{}]@{}",
                    correction.pauli,
                    correction.control,
                    Coord2(correction.target)
                )?;
            }
        }
        Op::Depolarize1 {
            probability,
            qubits,
        } => write_noise_op(f, "DEPOLARIZE1", *probability, qubits)?,
        Op::Depolarize2 {
            probability,
            qubits,
        } => write_noise_op(f, "DEPOLARIZE2", *probability, qubits)?,
        Op::PauliError {
            probability,
            pauli,
            qubits,
        } => write_noise_op(f, pauli_error_name(*pauli), *probability, qubits)?,
    }
    Ok(())
}

fn write_noise_op(
    f: &mut String,
    name: &str,
    probability: f64,
    qubits: &[glam::IVec2],
) -> fmt::Result {
    write!(f, "{name}({probability})")?;
    for &qubit in qubits {
        write!(f, " {}", Coord2(qubit))?;
    }
    Ok(())
}

fn pauli_error_name(pauli: PauliBasis) -> &'static str {
    match pauli {
        PauliBasis::X => "X_ERROR",
        PauliBasis::Y => "Y_ERROR",
        PauliBasis::Z => "Z_ERROR",
    }
}

fn write_flow(f: &mut String, flow: &Flow) -> fmt::Result {
    write!(f, "flow ")?;
    write_pauli_map(f, &flow.start)?;
    write!(f, " -> ")?;
    write_pauli_map(f, &flow.end)?;
    if !flow.measurements.is_empty() {
        write!(f, " meas ")?;
        write_separated(f, "*", &flow.measurements, |f, m| write!(f, "m{m}"))?;
    }
    if flow.sign {
        write!(f, " sign -1")?;
    }
    if let Some(center) = flow.center {
        write!(f, " center {}", Coord2(center))?;
    }
    match flow.marker {
        FlowMarker::Detector => {}
        FlowMarker::Discard => write!(f, " discard")?,
        FlowMarker::Restart => write!(f, " restart")?,
    }
    Ok(())
}

// ==============================================================================
// Graph levels
// ==============================================================================

fn write_level(f: &mut String, level: &SubGraph, depth: usize) -> fmt::Result {
    for (id, node) in level.nodes() {
        write_node(f, id, node, depth)?;
    }
    for edge in level.edges() {
        write_edge(f, edge, depth)?;
    }
    let pad = Indent(depth);
    if let Some(output) = level.value_output() {
        write!(f, "{pad}result ")?;
        write_value_ref(f, output)?;
        writeln!(f)?;
    }
    if !level.boundary_outputs().is_empty() {
        write!(f, "{pad}bindings")?;
        for output in level.boundary_outputs() {
            write!(f, " n{}", output.0)?;
        }
        writeln!(f)?;
    }
    Ok(())
}

fn write_node(f: &mut String, id: BloqNodeId, node: &BloqNode, depth: usize) -> fmt::Result {
    let pad = Indent(depth);
    match &node.kind {
        crate::BloqNodeKind::Quantum(quantum) => {
            writeln!(f, "{pad}n{} quantum {{", id.0)?;
            if let Some(timeline) = &quantum.timeline {
                write!(f, "{pad}  timeline rounds")?;
                for end in &timeline.layer_round_ends {
                    write!(f, " {end}")?;
                }
                writeln!(f)?;
            }
            for instance in &quantum.instances {
                write!(
                    f,
                    "{pad}  instance i{} t{} @ {}",
                    instance.id.0,
                    instance.template_id.0,
                    Coord2(instance.offset)
                )?;
                match instance.provenance {
                    crate::InstanceProvenance::Block { source } => {
                        write!(f, " from block {}", Coord3(source))?
                    }
                    crate::InstanceProvenance::Pipe { src, dst } => {
                        write!(f, " from pipe {} {}", Coord3(src), Coord3(dst))?
                    }
                    _ => {}
                }
                if let crate::InstanceProvenance::SpatialPortSubstitution { source, role, part } =
                    instance.provenance
                {
                    let role = role.as_str();
                    let part = match part {
                        crate::SpatialPortPart::Cube => "cube",
                        crate::SpatialPortPart::TemporalPort => "temporal-port",
                    };
                    write!(f, " from spatial-port {role} {part} {}", Coord3(source))?;
                }
                writeln!(f)?;
            }
            for detector in &quantum.detectors {
                write!(f, "{pad}  detector ")?;
                write_node_parity(f, &detector.parity)?;
                write_detector_coords(f, detector.coords.as_deref())?;
                writeln!(f)?;
            }
            for bundle_use in &quantum.detector_bundles {
                write!(f, "{pad}  use b{}", bundle_use.bundle.0)?;
                for instance in &bundle_use.instances {
                    write!(f, " i{}", instance.0)?;
                }
                writeln!(f, " @ {}", Coord2(bundle_use.offset))?;
            }
            for restart in &quantum.restarts {
                write!(f, "{pad}  restart ")?;
                write_node_parity(f, &restart.parity)?;
                writeln!(f)?;
            }
            for guard in &quantum.guards {
                write!(f, "{pad}  guard {}", guard.input)?;
                for instance in &guard.instances {
                    write!(f, " i{}", instance.0)?;
                }
                for detector in &guard.detectors {
                    write!(f, " d{detector}")?;
                }
                for bundle_use in &guard.detector_bundles {
                    write!(f, " b{bundle_use}")?;
                }
                for restart in &guard.restarts {
                    write!(f, " r{restart}")?;
                }
                for (index, parity) in &guard.detector_parities {
                    write!(f, " xd{index}=")?;
                    write_node_parity(f, parity)?;
                }
                for (index, parity) in &guard.restart_parities {
                    write!(f, " xr{index}=")?;
                    write_node_parity(f, parity)?;
                }
                writeln!(f)?;
            }
            if let Some(provenance) = provenance_text(&node.provenance) {
                writeln!(f, "{pad}  from {provenance}")?;
            }
            writeln!(f, "{pad}}}")?;
        }
        crate::BloqNodeKind::Classical(classical) => {
            write!(f, "{pad}n{} ", id.0)?;
            write_classical(f, classical)?;
            if let Some(slot) = node.activation {
                write!(f, " when v{slot}")?;
            }
            if let Some(provenance) = provenance_text(&node.provenance) {
                write!(f, " from {provenance}")?;
            }
            writeln!(f)?;
        }
        crate::BloqNodeKind::Region(region) => {
            write_region(f, id, region, &node.provenance, node.activation, depth)?;
        }
    }
    Ok(())
}

fn write_classical(f: &mut String, node: &ClassicalNode) -> fmt::Result {
    match node {
        ClassicalNode::Compute { expr } => {
            write!(f, "compute ")?;
            write_expr(f, expr)?;
        }
        ClassicalNode::Observable {
            index,
            measurements,
            operators,
        } => {
            write!(f, "observable ")?;
            match index {
                Some(index) => write!(f, "{index}")?,
                None => write!(f, "fragment")?,
            }
            if !measurements.is_empty() {
                write!(f, " measurements")?;
                write_classical_measurements(f, measurements)?;
            }
            if !operators.is_empty() {
                write!(f, " operators")?;
                write_classical_operators(f, operators)?;
            }
        }
        ClassicalNode::Discard { condition } => {
            write!(f, "discard ")?;
            write_expr(f, condition)?;
        }
    }
    Ok(())
}

fn write_classical_measurements(
    f: &mut String,
    measurements: &[InstanceMeasurement],
) -> fmt::Result {
    if !measurements.is_empty() {
        write!(f, " ")?;
        write_separated(f, "*", measurements, |f, m| {
            write_instance_measurement(f, *m)
        })?;
    }
    Ok(())
}

fn write_classical_operators(
    f: &mut String,
    operators: &[InstanceBoundaryOperator],
) -> fmt::Result {
    // Glue commas to the preceding Pauli-map token for whitespace tokenization.
    for (index, operator) in operators.iter().enumerate() {
        let face = match operator.face {
            crate::BoundaryFace::Input => "input",
            crate::BoundaryFace::Output => "output",
        };
        let sep = if index > 0 { "," } else { "" };
        write!(f, "{sep} i{} {face} ", operator.instance.0)?;
        write_pauli_map(f, &operator.operator)?;
    }
    Ok(())
}

fn write_region(
    f: &mut String,
    id: BloqNodeId,
    region: &RegionNode,
    provenance: &NodeProvenance,
    activation: Option<u32>,
    depth: usize,
) -> fmt::Result {
    let pad = Indent(depth);
    write!(f, "{pad}n{} ", id.0)?;
    let RegionNode::RepeatUntilSuccess {
        restart_condition,
        restart_source,
        ..
    } = region;
    write!(f, "rus ")?;
    write_expr(f, restart_condition)?;
    if let Some(source) = restart_source {
        write!(f, " source ")?;
        write_value_ref(f, *source)?;
    }
    if let Some(slot) = activation {
        write!(f, " when v{slot}")?;
    }
    writeln!(f, " {{")?;
    if let Some(provenance) = provenance_text(provenance) {
        writeln!(f, "{pad}  from {provenance}")?;
    }
    for (selector, body) in region.bodies() {
        writeln!(f, "{pad}  {} {{", selector.name())?;
        write_level(f, body, depth + 2)?;
        writeln!(f, "{pad}  }}")?;
    }
    writeln!(f, "{pad}}}")?;
    Ok(())
}

fn write_edge(f: &mut String, edge: BloqEdgeRef<'_>, depth: usize) -> fmt::Result {
    let pad = Indent(depth);
    write!(f, "{pad}n{} -> n{} ", edge.source.0, edge.target.0)?;
    match edge.edge {
        BloqEdge::Order => write!(f, "order")?,
        BloqEdge::Value { slot, role, output } => {
            write!(f, "value {slot}")?;
            if *output == ObservableOutput::Flip {
                write!(f, " flip")?;
            }
            write_role(f, role)?;
        }
        BloqEdge::Compose { slot, role } => {
            write!(f, "compose {slot}")?;
            write_role(f, role)?;
        }
        BloqEdge::Quantum(edge) => {
            write!(f, "quantum")?;
            for seam in &edge.pipes {
                write!(f, " ")?;
                write_pipe(f, &seam.pipe)?;
                if let Some(padding) = &seam.padding {
                    write!(
                        f,
                        " padding offset {} one t{} loop t{}",
                        Coord2(padding.offset),
                        padding.one_round.0,
                        padding.looped.0,
                    )?;
                }
            }
            if let Some(guard) = edge.guard {
                write!(f, " when ")?;
                write_value_ref(f, guard)?;
            }
        }
    }
    writeln!(f)
}

fn write_value_ref(f: &mut String, value: ValueRef) -> fmt::Result {
    write!(f, "n{}", value.node.0)?;
    if value.output == ObservableOutput::Flip {
        write!(f, ":flip")?;
    }
    Ok(())
}

fn write_role(f: &mut String, role: &ValueRole) -> fmt::Result {
    match role {
        ValueRole::Data => Ok(()),
        ValueRole::FeedbackFold { action } => write!(f, " feedback {action}"),
        ValueRole::ReadoutFold => write!(f, " readout"),
    }
}

// ==============================================================================
// Leaf renderers
// ==============================================================================

fn provenance_text(provenance: &NodeProvenance) -> Option<String> {
    let mut out = String::new();
    match provenance {
        NodeProvenance::None => return None,
        NodeProvenance::BlockComponent { members } => {
            out.push_str("blocks");
            for member in members {
                let _ = write!(out, " {}", Coord3(member.pos));
            }
        }
        NodeProvenance::TemporalPipe { pipe } => {
            out.push_str("pipe ");
            let _ = write_pipe(&mut out, pipe);
        }
        NodeProvenance::SpatialPortSubstitution { source, role } => {
            let _ = write!(out, "spatial-port {} {}", role.as_str(), Coord3(*source));
        }
        NodeProvenance::MemoryPadding { pipe, rounds } => {
            out.push_str("padding ");
            let _ = write_pipe(&mut out, pipe);
            let _ = write!(out, " rounds {rounds}");
        }
        NodeProvenance::Generator { ordinal } => {
            let _ = write!(out, "generator {ordinal}");
        }
        NodeProvenance::Action { ordinal } => {
            let _ = write!(out, "action {ordinal}");
        }
        NodeProvenance::BranchSelector { name } => {
            let _ = write!(out, "selector {}", encode_metadata_token(name));
        }
        NodeProvenance::OutputFrame { port, basis } => {
            let basis = match basis {
                Basis::X => "x",
                Basis::Z => "z",
            };
            let _ = write!(out, "frame {basis} {}", Coord3(*port));
        }
    }
    Some(out)
}

fn write_pipe(f: &mut String, pipe: &crate::TemporalPipeRef) -> fmt::Result {
    write!(f, "{}>{}", Coord3(pipe.src), Coord3(pipe.dst))?;
    if pipe.hadamard {
        write!(f, "h")?;
    }
    Ok(())
}

/// Write `items` with `sep` between consecutive entries.
fn write_separated<T>(
    f: &mut String,
    sep: &str,
    items: impl IntoIterator<Item = T>,
    mut each: impl FnMut(&mut String, T) -> fmt::Result,
) -> fmt::Result {
    for (index, item) in items.into_iter().enumerate() {
        if index > 0 {
            write!(f, "{sep}")?;
        }
        each(f, item)?;
    }
    Ok(())
}

fn write_pauli_map(f: &mut String, map: &PauliMap) -> fmt::Result {
    if map.is_empty() {
        return write!(f, "_");
    }
    write_separated(f, "*", map.iter(), |f, (coord, pauli)| {
        write!(f, "{pauli}{}", Coord2(*coord))
    })
}

fn write_template_parity(f: &mut String, parity: &DetectorParity<u32>) -> fmt::Result {
    write_parity(f, parity, |f, measurement| write!(f, "m{measurement}"))
}

fn write_node_parity(f: &mut String, parity: &DetectorParity<InstanceMeasurement>) -> fmt::Result {
    write_parity(f, parity, |f, measurement| {
        write_instance_measurement(f, measurement)
    })
}

fn write_bundle_parity(f: &mut String, parity: &DetectorParity<BundleMeasurement>) -> fmt::Result {
    write_parity(f, parity, |f, measurement| {
        write!(f, "o{}:m{}", measurement.owner, measurement.measurement)
    })
}

fn write_parity<M: Copy + Ord>(
    f: &mut String,
    parity: &DetectorParity<M>,
    mut measurement: impl FnMut(&mut String, M) -> fmt::Result,
) -> fmt::Result {
    if parity.sign() {
        write!(f, "-")?;
    }
    if parity.terms().is_empty() {
        return write!(f, "0");
    }
    write_separated(f, "*", parity.terms(), |f, term| match term {
        DetectorTerm::Measurement(m) => measurement(f, *m),
        DetectorTerm::LoopState(state) => write!(f, "s{}", state.0),
    })
}

fn write_instance_measurement(f: &mut String, measurement: InstanceMeasurement) -> fmt::Result {
    write!(
        f,
        "i{}:m{}",
        measurement.instance.0, measurement.measurement
    )
}

fn write_detector_coords(f: &mut String, coords: Option<&[f64]>) -> fmt::Result {
    let Some(coords) = coords else {
        return Ok(());
    };
    write!(f, " @ (")?;
    write_separated(f, ",", coords, |f, coord| write!(f, "{coord}"))?;
    write!(f, ")")
}

/// Precedence levels: `|` = 0, `^` = 1, `&` = 2, `!` = 3, atoms = 4. A child
/// below `min` is parenthesized. Left children pass their operator's own
/// level (flat chains stay bare); right children pass one higher, so an
/// equal-precedence right subtree keeps its parentheses and the tree shape
/// round-trips exactly, not just its semantics.
fn write_expr(f: &mut String, expr: &ClassicalExpr) -> fmt::Result {
    write_expr_at(f, expr, 0)
}

fn write_expr_at(f: &mut String, expr: &ClassicalExpr, min: u8) -> fmt::Result {
    let prec = precedence(expr);
    if prec < min {
        write!(f, "(")?;
        write_expr_at(f, expr, 0)?;
        return write!(f, ")");
    }
    match expr {
        ClassicalExpr::In(slot) => write!(f, "in{slot}"),
        ClassicalExpr::Const(bit) => write!(f, "{}", u8::from(*bit)),
        ClassicalExpr::Parity { inputs, constant } => {
            write!(f, "parity({}", u8::from(*constant))?;
            for slot in inputs {
                write!(f, ", in{slot}")?;
            }
            write!(f, ")")
        }
        ClassicalExpr::Not(inner) => {
            write!(f, "!")?;
            write_expr_at(f, inner, 3)
        }
        ClassicalExpr::Or(operands) => write_operands(f, operands, "or", "|", 0),
        ClassicalExpr::Xor(operands) => write_operands(f, operands, "xor", "^", 1),
        ClassicalExpr::And(operands) => write_operands(f, operands, "and", "&", 2),
        ClassicalExpr::Select(operands) => write_function(f, "select", operands.as_ref()),
    }
}

fn write_operands(
    f: &mut String,
    operands: &[ClassicalExpr],
    name: &str,
    op: &str,
    prec: u8,
) -> fmt::Result {
    if operands.len() < 2 {
        return write_function(f, name, operands);
    }
    for (index, operand) in operands.iter().enumerate() {
        if index != 0 {
            write!(f, " {op} ")?;
        }
        // Preserve explicit same-operator nesting on either side.
        write_expr_at(f, operand, prec + 1)?;
    }
    Ok(())
}

fn write_function(f: &mut String, name: &str, operands: &[ClassicalExpr]) -> fmt::Result {
    write!(f, "{name}(")?;
    for (index, operand) in operands.iter().enumerate() {
        if index != 0 {
            write!(f, ", ")?;
        }
        write_expr(f, operand)?;
    }
    write!(f, ")")
}

fn precedence(expr: &ClassicalExpr) -> u8 {
    match expr {
        ClassicalExpr::Or(..) => 0,
        ClassicalExpr::Xor(..) => 1,
        ClassicalExpr::And(..) => 2,
        ClassicalExpr::Not(..) => 3,
        ClassicalExpr::In(_)
        | ClassicalExpr::Const(_)
        | ClassicalExpr::Select(_)
        | ClassicalExpr::Parity { .. } => 4,
    }
}

struct Coord2(IVec2);

impl fmt::Display for Coord2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({},{})", self.0.x, self.0.y)
    }
}

struct Coord3(IVec3);

impl fmt::Display for Coord3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({},{},{})", self.0.x, self.0.y, self.0.z)
    }
}

struct Indent(usize);

impl fmt::Display for Indent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for _ in 0..self.0 {
            f.write_str("  ")?;
        }
        Ok(())
    }
}
