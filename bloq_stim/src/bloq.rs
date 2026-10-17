use crate::emit::{
    ResolvedLoopStates, StimAnnotationScope, StimCircuitAnnotations, StimDetector,
    StimEmissionError, StimEmitContext, StimEmitOptions, StimMeasurementMapper, StimRepeatState,
    emit_stim_circuit, first_unsupported_gate,
};
use crate::layout::{QubitLayout, coordinate_index, detector_index};
use crate::measurement_frame::{MeasurementFrame, stim_record_lookback};
use crate::text_utils::push_int;
use bloq_circuit::{
    Basis, BodyId, CircuitError, CoordCircuit, FlowMarker, GateType, LoopStateId, NoiseModel, Op,
    Pauli, PauliMap,
};
use bloq_ir::{
    Bloq, BloqNode, BloqNodeId, BloqNodeKind, BodySelector, ClassicalAssignment, ClassicalNode,
    LevelPath, MomentLane, RegionNode, SubGraph, TemplateId, ValidatedPlans, align_moment_lanes,
    aligned_moment_segments,
    lowering::{
        InstanceMeasurement, InstantiationOptions, NodeEmissionPlan, TemplateDetectorScope,
        TemplateInstanceId,
    },
};
use glam::IVec2;
use rustc_hash::{FxHashMap, FxHashSet};
use std::{
    borrow::Cow,
    collections::{BTreeMap, BTreeSet},
};

/// One output-frame correction recipe for an isolated T attempt.
/// The corrected bit XORs measurement parity, `sign`, and the GAP decoder flip.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IsolatedTFrameRecipe {
    /// Physical measurement columns XORed into the raw frame bit.
    pub measurements: Vec<u32>,
    /// Constant XORed with the measurement parity before decoder correction.
    pub sign: bool,
    /// GAP observable whose decoder flip corrects this frame bit.
    pub gap_observable: u32,
}

/// One terminal stabilizer sheet appended to the causal companion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IsolatedTFrontierSheet {
    /// Observable id this sheet uses in [`IsolatedTAttemptArtifacts::sheets`].
    pub observable: u32,
    /// Homogeneous stabilizer basis of the terminal boundary operator.
    pub basis: Basis,
    /// Expected parity sign of the boundary flow.
    pub sign: bool,
}

/// Index data shared by all isolated-attempt circuit variants.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IsolatedTAttemptManifest {
    /// Path of the emitted `RepeatUntilSuccess` body.
    pub rus_path: LevelPath,
    /// Expected sign of every emitted detector, in circuit detector order.
    pub detector_signs: Vec<bool>,
    /// Zero-based detector indices tagged `POST-SELECTION`.
    pub postselection_detectors: Vec<u32>,
    /// GAP observable carrying the logical X sheet.
    pub gap_x_observable: u32,
    /// GAP observable carrying the logical Z sheet.
    pub gap_z_observable: u32,
    /// Terminal output X-frame correction recipe.
    pub frame_x: IsolatedTFrameRecipe,
    /// Terminal output Z-frame correction recipe.
    pub frame_z: IsolatedTFrameRecipe,
    /// Frontier sheets in append order. Sheet `i` maps to synthetic detector
    /// `base_companion_detector_count + i` in the causal DEM transform.
    pub frontier_sheets: Vec<IsolatedTFrontierSheet>,
    /// Static signs multiplying direct X/Y/Z `EXP_VAL` probes.
    pub exp_val_signs: [i8; 3],
}

/// Physical sampling circuits and their shared causal decoding companion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IsolatedTAttemptArtifacts {
    /// Full-noise S control, with direct X/Y/Z `EXP_VAL` probes.
    pub physical_s: String,
    /// Full-noise honest-T circuit, with direct X/Y/Z `EXP_VAL` probes.
    pub physical_t: String,
    /// Partially-noiseless S-proxy companion carrying the GAP pair.
    pub companion: String,
    /// Terminal stabilizer sheet observables, as a standalone suffix.
    ///
    /// Appending this to [`Self::companion`] gives the sheeted companion.
    /// It is kept separate because consumers need the suffix *alone* (to
    /// build the frontier DEM), and recovering it from a concatenated field
    /// means slicing one circuit's text by another's byte length.
    pub sheets: String,
    /// Detector, frame, GAP, frontier, and probe index data.
    pub manifest: IsolatedTAttemptManifest,
}

/// Whether an emission entry point checks its input program first.
///
/// Emission needs the WF-9 per-node merge either way (it is what the backend
/// emits from); this only selects whether the full T2 well-formedness audit
/// runs alongside it. The default is [`Trusted`](Self::Trusted), so ordinary
/// emission does not run that audit.
///
/// Use [`Checked`](Self::Checked) to request the audit before emitting. In
/// trusted mode, malformed input can produce a wrong circuit, although plan
/// construction and emission still report errors they encounter.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum InputTrust {
    /// Skip the full T2 audit; still build the required emission plans.
    #[default]
    Trusted,
    /// Run the full T2 audit before emitting.
    Checked,
}

/// Options for noise, input validation, and whole-program moment alignment.
///
/// Alignment rejects noise and segmented output with
/// [`StimEmissionError::AlignMomentsWithNoise`] and
/// [`StimEmissionError::AlignMomentsWithSegments`].
///
/// # Examples
///
/// ```
/// use bloq_stim::{BloqStimOptions, InputTrust, emit_bloq_stim_with};
///
/// let graph = bloq_graph::GalleryItem::CNOT.build();
/// let program = bloq_compile::compile(&graph, 3)?;
/// let noise = bloq_circuit::NoiseModel::uniform_depolarizing(1e-3);
///
/// let options = BloqStimOptions::new()
///     .with_noise(&noise)
///     .with_trust(InputTrust::Checked);
/// assert!(emit_bloq_stim_with(&program, &options)?.contains("DEPOLARIZE1"));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct BloqStimOptions<'a> {
    noise: Option<&'a NoiseModel>,
    trust: InputTrust,
    align_moments: bool,
}

impl<'a> BloqStimOptions<'a> {
    /// Noiseless emission without a full input audit, as [`emit_bloq_stim`] uses.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Annotate every instantiated circuit with `noise` before emitting.
    #[must_use]
    pub fn with_noise(mut self, noise: &'a NoiseModel) -> Self {
        self.noise = Some(noise);
        self
    }

    /// Whether to run the full T2 audit first; see [`InputTrust`].
    #[must_use]
    pub fn with_trust(mut self, trust: InputTrust) -> Self {
        self.trust = trust;
        self
    }

    /// Align compatible node moments across each block layer.
    #[must_use]
    pub fn with_align_moments(mut self, align: bool) -> Self {
        self.align_moments = align;
        self
    }
}

/// Linearizes a [`Bloq`] into Stim text, noiseless and trusting the program to
/// be well-formed ([`InputTrust::Trusted`]).
///
/// Emits node-local circuits in deterministic topological order and resolves
/// node-local annotations against one global measurement frame. Use
/// [`emit_bloq_stim_with`] for noise or a checked input.
///
/// # Errors
///
/// Returns [`StimEmissionError`] if the program cannot be statically emitted.
pub fn emit_bloq_stim(program: &Bloq) -> Result<String, StimEmissionError> {
    emit_bloq_stim_with(program, &BloqStimOptions::new())
}

/// [`emit_bloq_stim`] under explicit [`BloqStimOptions`].
///
/// With moment alignment enabled, repeat bodies are flattened and compatible raw moments from
/// quantum nodes occupying the same source layer share global `TICK` slots.
///
/// # Errors
///
/// Returns [`StimEmissionError`] for invalid, unsupported, or inconsistent input.
pub fn emit_bloq_stim_with(
    program: &Bloq,
    options: &BloqStimOptions<'_>,
) -> Result<String, StimEmissionError> {
    if options.align_moments {
        if options.noise.is_some() {
            return Err(StimEmissionError::AlignMomentsWithNoise);
        }
        if options.trust == InputTrust::Checked {
            program.validate()?;
        }
        reject_unsupported(program)?;
        let mut flat = program.clone();
        flat.flatten()?;
        let plans = flat.emission_plans(&InstantiationOptions::default())?;
        let order = flat.deterministic_emit_order()?;
        return BloqStimEmitter::new(&flat, None, &plans).emit_order_aligned(&order);
    }
    let (plans, order) = prepare_untransformed(program, options)?;
    BloqStimEmitter::new(program, options.noise, &plans).emit_order_untransformed(&order)
}

/// Shared plans and order for whole-text and segmented emission.
///
/// Checked inputs return validation errors before capability errors. Trusted
/// inputs skip the audit, so unsupported constructs must be rejected before
/// merge work.
fn prepare_untransformed(
    program: &Bloq,
    options: &BloqStimOptions<'_>,
) -> Result<(ValidatedPlans, Vec<BloqNodeId>), StimEmissionError> {
    if options.trust == InputTrust::Trusted {
        reject_unsupported(program)?;
    }
    let plans = emission_plans(program, options.noise, options.trust)?;
    if options.trust == InputTrust::Checked {
        reject_unsupported(program)?;
    }
    Ok((plans, program.deterministic_emit_order()?))
}

/// Emit one flattened attempt of the unique isolated-T RUS body.
///
/// The physical strings contain no `OBSERVABLE_INCLUDE`; they end in direct
/// X/Y/Z `EXP_VAL` probes for Clifft. The companion strings are Stim/DEM
/// circuits: `companion` carries the X/Z GAP pair, and `sheets` is a suffix
/// carrying one observable per terminal stabilizer frontier, which appended
/// to `companion` gives the sheeted circuit.
/// Memory latency must already be inserted into `program`.
///
/// # Errors
///
/// Returns [`StimEmissionError`] if the program is not one valid isolated-T attempt.
pub fn emit_isolated_t_attempts(
    program: &Bloq,
    physical_error_probability: f64,
) -> Result<IsolatedTAttemptArtifacts, StimEmissionError> {
    if !physical_error_probability.is_finite() || !(0.0..=1.0).contains(&physical_error_probability)
    {
        return Err(StimEmissionError::InvalidNoiseProbability);
    }

    let mut flat = program.clone();
    flat.flatten()?;
    // The flattened clone is this function's own construction, not caller
    // input, so it inherits the caller's trust in `program`.
    let plans = emission_plans(&flat, None, InputTrust::default())?;
    let path = isolated_rus_path(&flat)?;
    let body = flat
        .level_at(&path)
        .ok_or(StimEmissionError::MalformedGraph(
            "isolated-T RUS body path is stale",
        ))?;
    let order = body.deterministic_emit_order()?;
    let cultivation = isolated_cultivation_node(&flat, body)?;
    let terminal = order
        .iter()
        .rev()
        .copied()
        .find(|&node| body[node].try_quantum().is_some())
        .ok_or(StimEmissionError::MalformedGraph(
            "isolated-T RUS body has no quantum node",
        ))?;
    let (gap_x, gap_z, gap_records) = isolated_gap_observables(body)?;
    let frames = flat.output_frames();
    if frames.len() != 1 || flat.logical_outputs().len() != 1 {
        return Err(StimEmissionError::MalformedGraph(
            "isolated-T program must have exactly one output frame and logical output",
        ));
    }

    let noise = NoiseModel::uniform_depolarizing(physical_error_probability);
    let render = |mode| {
        BloqStimEmitter::new_attempt(&flat, body, path.clone(), &noise, mode, &plans)
            .emit_order_attempt(&order)
    };
    let mut physical_s = render(AttemptMode::SControl)?;
    let mut physical_t = render(AttemptMode::HonestT)?;
    let companion = render(AttemptMode::Companion { cultivation })?;
    ensure_attempt_alignment(&physical_s, &physical_t, &companion)?;

    let frame_x = isolated_frame_recipe(
        flat.top(),
        frames[0].x,
        &companion.measurement_columns,
        &gap_records,
    )?;
    let frame_z = isolated_frame_recipe(
        flat.top(),
        frames[0].z,
        &companion.measurement_columns,
        &gap_records,
    )?;
    if frame_x.gap_observable != gap_z || frame_z.gap_observable != gap_x {
        return Err(StimEmissionError::MalformedGraph(
            "isolated-T output frames do not cross-map X to GAP-Z and Z to GAP-X",
        ));
    }

    let exp_val_signs = append_direct_exp_vals(
        &mut physical_s.text,
        &flat,
        &flat.logical_outputs()[0].x,
        &flat.logical_outputs()[0].z,
    )?;
    let t_signs = append_direct_exp_vals(
        &mut physical_t.text,
        &flat,
        &flat.logical_outputs()[0].x,
        &flat.logical_outputs()[0].z,
    )?;
    if t_signs != exp_val_signs {
        return Err(StimEmissionError::MalformedGraph(
            "isolated-T S/T probe signs disagree",
        ));
    }

    let mut sheets = String::new();
    let frontier_sheets = append_frontier_sheets(
        &mut sheets,
        &flat,
        body,
        terminal,
        &companion.measurement_columns,
    )?;
    let detector_signs = companion
        .detectors
        .iter()
        .map(|detector| detector.sign)
        .collect();
    let postselection_detectors = companion
        .detectors
        .iter()
        .enumerate()
        .filter(|(_, detector)| detector.postselection)
        .map(|(index, _)| detector_index(index))
        .collect::<Result<_, _>>()?;

    Ok(IsolatedTAttemptArtifacts {
        physical_s: physical_s.text,
        physical_t: physical_t.text,
        companion: companion.text,
        sheets,
        manifest: IsolatedTAttemptManifest {
            rus_path: path,
            detector_signs,
            postselection_detectors,
            gap_x_observable: gap_x,
            gap_z_observable: gap_z,
            frame_x,
            frame_z,
            frontier_sheets,
            exp_val_signs,
        },
    })
}

fn isolated_rus_path(program: &Bloq) -> Result<LevelPath, StimEmissionError> {
    let mut rus = program.nodes().filter_map(|(id, node)| {
        matches!(
            node.try_region(),
            Some(RegionNode::RepeatUntilSuccess { .. })
        )
        .then_some(id)
    });
    let id = rus.next().ok_or(StimEmissionError::MalformedGraph(
        "isolated-T program has no top-level RepeatUntilSuccess",
    ))?;
    if rus.next().is_some() {
        return Err(StimEmissionError::MalformedGraph(
            "isolated-T program has more than one top-level RepeatUntilSuccess",
        ));
    }
    Ok(LevelPath::default().child(id, BodySelector::Body))
}

fn isolated_cultivation_node(
    program: &Bloq,
    body: &SubGraph,
) -> Result<BloqNodeId, StimEmissionError> {
    let mut cultivation = None;
    let mut quantum_count = 0;
    for (node, quantum) in body.quantum_nodes() {
        quantum_count += 1;
        if quantum.instances.len() != 1 {
            return Err(StimEmissionError::MalformedGraph(
                "isolated-T body quantum nodes must have one template instance",
            ));
        }
        let template = program
            .templates()
            .get(quantum.instances[0].template_id)
            .ok_or(StimEmissionError::MalformedGraph(
                "isolated-T body instance references a missing template",
            ))?;
        if first_unsupported_gate(&template.circuit).is_some()
            && cultivation.replace(node).is_some()
        {
            return Err(StimEmissionError::MalformedGraph(
                "isolated-T body has more than one non-Clifford stage",
            ));
        }
    }
    if quantum_count < 2 {
        return Err(StimEmissionError::MalformedGraph(
            "isolated-T body must contain cultivation and escape stages",
        ));
    }
    cultivation.ok_or(StimEmissionError::MalformedGraph(
        "isolated-T body has no non-Clifford cultivation stage",
    ))
}

type GapRecordSets = BTreeMap<u32, BTreeSet<InstanceMeasurement>>;

fn isolated_gap_observables(
    body: &SubGraph,
) -> Result<(u32, u32, GapRecordSets), StimEmissionError> {
    let mut indices = BTreeSet::new();
    for (_, node) in body.nodes() {
        if let Some(ClassicalNode::Observable {
            index: Some(index), ..
        }) = node.try_classical()
        {
            indices.insert(*index);
        }
    }
    if indices.len() != 2 {
        return Err(StimEmissionError::MalformedGraph(
            "isolated-T body must contain two GAP observables",
        ));
    }
    let mut gap_x = None;
    let mut gap_z = None;
    let mut records = BTreeMap::new();
    for index in indices {
        let observable = observable_node(body, index)?;
        let measurements = body
            .resolve_classical(observable, ClassicalAssignment::Uniform(false))?
            .measurements;
        records.insert(index, measurements);
        match observable_boundary_basis(body, observable)? {
            Basis::X if gap_x.replace(index).is_none() => {}
            Basis::Z if gap_z.replace(index).is_none() => {}
            _ => {
                return Err(StimEmissionError::MalformedGraph(
                    "isolated-T GAP observables must have one X and one Z sheet",
                ));
            }
        }
    }
    Ok((
        gap_x.expect("two checked GAP bases include X"),
        gap_z.expect("two checked GAP bases include Z"),
        records,
    ))
}

fn observable_node(level: &SubGraph, index: u32) -> Result<BloqNodeId, StimEmissionError> {
    let mut nodes = level.nodes().filter_map(|(id, node)| {
        matches!(
            node.try_classical(),
            Some(ClassicalNode::Observable { index: Some(candidate), .. }) if *candidate == index
        )
        .then_some(id)
    });
    let node = nodes.next().ok_or(StimEmissionError::MalformedGraph(
        "decoder references a missing isolated-T observable",
    ))?;
    if nodes.next().is_some() {
        return Err(StimEmissionError::MalformedGraph(
            "isolated-T graph repeats an observable index in one level",
        ));
    }
    Ok(node)
}

fn observable_boundary_basis(
    level: &SubGraph,
    observable: BloqNodeId,
) -> Result<Basis, StimEmissionError> {
    let mut basis = None;
    let mut stack = vec![observable];
    let mut visited = FxHashSet::default();
    while let Some(producer) = stack.pop() {
        if !visited.insert(producer) {
            continue;
        }
        let Some(ClassicalNode::Observable { operators, .. }) = level[producer].try_classical()
        else {
            continue;
        };
        stack.extend(
            level
                .data_inputs(producer)
                .filter(|input| input.output.is_none())
                .map(|input| input.producer),
        );
        for operator in operators {
            for (_, pauli) in &operator.operator {
                let next = match pauli {
                    Pauli::X => Basis::X,
                    Pauli::Z => Basis::Z,
                    Pauli::I | Pauli::Y => {
                        return Err(StimEmissionError::MalformedGraph(
                            "isolated-T boundary sheet is not homogeneous X or Z",
                        ));
                    }
                };
                if basis.is_some_and(|current| current != next) {
                    return Err(StimEmissionError::MalformedGraph(
                        "isolated-T boundary sheet mixes X and Z support",
                    ));
                }
                basis = Some(next);
            }
        }
    }
    basis.ok_or(StimEmissionError::MalformedGraph(
        "isolated-T GAP observable has no boundary sheet",
    ))
}

fn isolated_frame_recipe(
    top: &SubGraph,
    frame: BloqNodeId,
    columns: &FxHashMap<InstanceMeasurement, u32>,
    gap_records: &GapRecordSets,
) -> Result<IsolatedTFrameRecipe, StimEmissionError> {
    let value = top.resolve_classical(frame, ClassicalAssignment::Uniform(false))?;
    let records = value.measurements;
    let gap_observable = gap_records
        .iter()
        .find_map(|(&index, candidate)| (candidate == &records).then_some(index))
        .ok_or(StimEmissionError::MalformedGraph(
            "isolated-T output frame records do not match either GAP observable",
        ))?;
    let mut measurements = records
        .iter()
        .map(|measurement| {
            columns
                .get(measurement)
                .copied()
                .ok_or(StimEmissionError::MalformedGraph(
                    "isolated-T frame references a measurement outside the RUS body",
                ))
        })
        .collect::<Result<Vec<_>, _>>()?;
    measurements.sort_unstable();
    Ok(IsolatedTFrameRecipe {
        measurements,
        sign: value.sign,
        gap_observable,
    })
}

fn ensure_attempt_alignment(
    physical_s: &AttemptStimRender,
    physical_t: &AttemptStimRender,
    companion: &AttemptStimRender,
) -> Result<(), StimEmissionError> {
    if physical_s.measurement_columns != physical_t.measurement_columns
        || physical_s.measurement_columns != companion.measurement_columns
        || physical_s.detectors != physical_t.detectors
        || physical_s.detectors != companion.detectors
    {
        return Err(StimEmissionError::MalformedGraph(
            "isolated-T S, T, and companion measurement/detector recipes disagree",
        ));
    }
    Ok(())
}

fn append_frontier_sheets(
    output: &mut String,
    program: &Bloq,
    body: &SubGraph,
    terminal: BloqNodeId,
    columns: &FxHashMap<InstanceMeasurement, u32>,
) -> Result<Vec<IsolatedTFrontierSheet>, StimEmissionError> {
    let quantum = body[terminal].expect_quantum();
    let instance = quantum
        .instances
        .first()
        .ok_or(StimEmissionError::MalformedGraph(
            "isolated-T terminal node has no template instance",
        ))?;
    let template =
        program
            .templates()
            .get(instance.template_id)
            .ok_or(StimEmissionError::MalformedGraph(
                "isolated-T terminal instance references a missing template",
            ))?;
    let measurement_count = columns
        .values()
        .copied()
        .max()
        .and_then(|last| last.checked_add(1))
        .ok_or(StimEmissionError::MalformedGraph(
            "isolated-T attempt emitted no measurements",
        ))?;
    let coords = program.sorted_layout_coords()?;
    let layout = QubitLayout::new(coordinate_index(coords.iter().copied())?)?;
    let max_observable = program
        .levels()
        .flat_map(|(_, level)| level.nodes())
        .filter_map(|(_, node)| match node.try_classical() {
            Some(ClassicalNode::Observable { index, .. }) => *index,
            _ => None,
        })
        .max();
    let mut next_observable = match max_observable {
        Some(index) => index
            .checked_add(1)
            .ok_or(StimEmissionError::MalformedGraph(
                "isolated-T observable id overflow",
            ))?,
        None => 0,
    };
    let mut sheets = Vec::new();
    for flow in template.boundary_flows.iter().filter(|flow| {
        flow.start.is_empty() && !flow.end.is_empty() && flow.marker == FlowMarker::Detector
    }) {
        let basis = homogeneous_basis(&flow.end)?;
        let mut lookbacks = flow
            .measurements
            .iter()
            .map(|&measurement| {
                let column = columns
                    .get(&InstanceMeasurement {
                        instance: instance.id,
                        measurement,
                    })
                    .copied()
                    .ok_or(StimEmissionError::MalformedGraph(
                        "terminal frontier references an unemitted measurement",
                    ))?;
                let lookback = measurement_count - column;
                stim_record_lookback(lookback).ok_or(StimEmissionError::MalformedGraph(
                    "terminal frontier measurement lookback exceeds Stim range",
                ))
            })
            .collect::<Result<Vec<_>, _>>()?;
        lookbacks.sort_unstable();
        crate::emit::emit_observable_include_records(output, next_observable, &lookbacks);
        let operator = flow.end.try_translated(instance.offset)?;
        crate::emit::emit_observable_include_pauli_targets(
            output,
            next_observable,
            &operator,
            &layout,
        )?;
        sheets.push(IsolatedTFrontierSheet {
            observable: next_observable,
            basis,
            sign: flow.sign,
        });
        next_observable =
            next_observable
                .checked_add(1)
                .ok_or(StimEmissionError::MalformedGraph(
                    "isolated-T observable id overflow",
                ))?;
    }
    if sheets.is_empty() {
        return Err(StimEmissionError::MalformedGraph(
            "isolated-T terminal node has no open detector frontier",
        ));
    }
    Ok(sheets)
}

fn homogeneous_basis(operator: &PauliMap) -> Result<Basis, StimEmissionError> {
    let mut basis = None;
    for (_, pauli) in operator {
        let next = match pauli {
            Pauli::X => Basis::X,
            Pauli::Z => Basis::Z,
            Pauli::I | Pauli::Y => {
                return Err(StimEmissionError::MalformedGraph(
                    "terminal frontier is not homogeneous X or Z",
                ));
            }
        };
        if basis.is_some_and(|current| current != next) {
            return Err(StimEmissionError::MalformedGraph(
                "terminal frontier mixes X and Z support",
            ));
        }
        basis = Some(next);
    }
    basis.ok_or(StimEmissionError::MalformedGraph(
        "terminal frontier has empty Pauli support",
    ))
}

fn append_direct_exp_vals(
    output: &mut String,
    program: &Bloq,
    logical_x: &PauliMap,
    logical_z: &PauliMap,
) -> Result<[i8; 3], StimEmissionError> {
    let coords = program.sorted_layout_coords()?;
    let layout = coordinate_index(coords.iter().copied())?;
    let (logical_y, y_sign) = signed_logical_y(logical_x, logical_z)?;
    for operator in [logical_x, &logical_y, logical_z] {
        output.push_str("EXP_VAL ");
        let mut first = true;
        for (coord, pauli) in operator {
            if !first {
                output.push('*');
            }
            first = false;
            output.push_str(&pauli.to_string());
            push_int(
                output,
                *layout.get(coord).ok_or(StimEmissionError::MalformedGraph(
                    "logical output probe names a qubit outside the global layout",
                ))?,
            );
        }
        if first {
            return Err(StimEmissionError::MalformedGraph(
                "logical output probe has empty Pauli support",
            ));
        }
        output.push('\n');
    }
    Ok([1, y_sign, 1])
}

fn signed_logical_y(
    logical_x: &PauliMap,
    logical_z: &PauliMap,
) -> Result<(PauliMap, i8), StimEmissionError> {
    // Phase exponent in powers of i. The leading i makes Y = iXZ Hermitian.
    let phase = (1 + logical_x.product_phase(logical_z)) % 4;
    let sign = match phase {
        0 => 1,
        2 => -1,
        _ => {
            return Err(StimEmissionError::MalformedGraph(
                "logical iXZ probe is not Hermitian",
            ));
        }
    };
    Ok((logical_x ^ logical_z, sign))
}

/// The per-node emission plans the merge pass built (IR-02) — the exact plans
/// [`BloqStimEmitter::emit_instantiated_node`] emits from, so the SEM-MERGE
/// runs once here rather than again per node.
///
/// `trust` selects whether the full T2 audit runs alongside the merge.
fn emission_plans(
    program: &Bloq,
    noise: Option<&NoiseModel>,
    trust: InputTrust,
) -> Result<ValidatedPlans, StimEmissionError> {
    let options = noise.map_or_else(InstantiationOptions::default, InstantiationOptions::noisy);
    if trust == InputTrust::Checked {
        return Ok(program.validate_with_plans(&options)?);
    }
    Ok(program.emission_plans(&options)?)
}

/// Reject a program the static Stim backend cannot emit, before any emission
/// work. Un-emittability of a gate or a node *kind* (runtime-control regions,
/// [`Discard`](ClassicalNode::Discard)) is a static property, so both are checked here rather than
/// mid-walk in [`BloqStimEmitter::emit_node`].
fn reject_unsupported(program: &Bloq) -> Result<(), StimEmissionError> {
    // Region nodes are rejected by name first: a T block's cultivation template
    // carries the real non-Clifford `T_DAG`, but the reason the static path
    // cannot emit it is the enclosing `RepeatUntilSuccess` region (dynamic
    // control), not the gate. Rejecting the gate here would leak an
    // implementation detail and mask the structural reason. A genuinely
    // top-level non-Clifford gate (no region) still falls through to the gate
    // check below.
    if let Some(node) = first_unsupported_program_node(program) {
        return Err(StimEmissionError::UnsupportedNode(node));
    }
    if let Some(gate) = first_unsupported_program_gate(program) {
        return Err(StimEmissionError::UnsupportedGate(gate));
    }
    Ok(())
}

/// The name of the first node the static backend cannot emit, in node-id order,
/// detected up front so no partial emission happens first.
fn first_unsupported_program_node(program: &Bloq) -> Option<&'static str> {
    program
        .nodes()
        .find_map(|(_, node)| stim_unsupported_reason(node))
        .or_else(|| {
            program
                .has_conditional_membership()
                .then_some("conditional quantum seams")
        })
}

/// Why the static backend cannot emit `node`, or `None` if it can. The single
/// source of truth for emittability: `reject_unsupported` scans it up front and
/// [`BloqStimEmitter::emit_node`] asserts against it, so the two never diverge.
fn stim_unsupported_reason(node: &BloqNode) -> Option<&'static str> {
    if node.activation.is_some() {
        return Some("activation (conditional execution)");
    }
    if node
        .try_quantum()
        .is_some_and(|quantum| !quantum.guards.is_empty())
    {
        return Some("conditional component membership");
    }
    match (node.try_classical(), node.try_region()) {
        // An Observable produces corrected parity; its static-path flip
        // estimate is identically false (no decoder in the loop — corrections
        // defer to the offline decoder / logical Pauli frame), so it reduces to
        // its records (SEM-DECODE), the same degenerate reduction
        // the noiseless executor uses.
        // `OBSERVABLE_INCLUDE` is a *linear* (XOR) accumulation of measurement
        // records, so a feedback whose condition is linear (`Xor`/`Not`/`Const`
        // over measurement bits) folds directly into the affected observable and
        // emits fine. A *nonlinear* condition (`And`/`Or`) is not XOR-expressible;
        // it needs conditional execution, so we reject it up front
        // rather than fold it incorrectly.
        // `Xor`/`Not`/`In`/`Const` are linear (a `Const` or `Not` only toggles
        // a sign Stim already ignores); see [`ClassicalExpr::is_linear`].
        (Some(ClassicalNode::Compute { expr }), _) if !expr.is_linear() => {
            Some("Compute (nonlinear And/Or condition requires conditional execution)")
        }
        // A `Discard` is shot postselection the static backend cannot express
        // (Stim has no reject-shot op), so reject it rather than silently drop it
        // (SEM-DISCARD).
        (Some(ClassicalNode::Discard { .. }), _) => Some("Discard (shot postselection)"),
        (_, Some(_)) => Some("RepeatUntilSuccess (postselected retry)"),
        _ => None,
    }
}

/// One node's slice of a segmented emission: its Stim text plus where that
/// text sits in the program's measurement record stream.
///
/// A node that emits nothing (most classical nodes) still gets a segment, with
/// empty `text` and a zero `measurement_count`, so the segment list always
/// mirrors [`Bloq::deterministic_emit_order`] one-for-one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StimSegment {
    /// The node this chunk was emitted from.
    pub node_id: BloqNodeId,
    /// The node's Stim text.
    pub text: String,
    /// Absolute index of this chunk's first measurement record in the
    /// reassembled circuit.
    pub measurement_start: usize,
    /// How many measurement records this chunk contributes.
    pub measurement_count: usize,
}

/// Emitted Stim text split into the shared header and one chunk per node.
///
/// Chunks share one global measurement frame, so a node's `rec[-k]` detector
/// targets cross chunk boundaries and only resolve once reassembled — see
/// [`Self::to_text`].
#[derive(Debug, Clone)]
pub struct BloqStimSegments {
    /// `QUBIT_COORDS` preamble shared by every node.
    pub header: String,
    /// Per-node chunks in [`Bloq::deterministic_emit_order`].
    pub segments: Vec<StimSegment>,
}

impl BloqStimSegments {
    /// Reassemble the header and every chunk into one circuit — the whole
    /// point of the split being reversible, and the only form in which the
    /// cross-chunk `rec[-k]` targets resolve.
    pub fn to_text(&self) -> String {
        let capacity = self.header.len()
            + self
                .segments
                .iter()
                .map(|segment| segment.text.len())
                .sum::<usize>();
        let mut text = String::with_capacity(capacity);
        text.push_str(&self.header);
        for segment in &self.segments {
            text.push_str(&segment.text);
        }
        text
    }

    /// How many measurement records the reassembled circuit emits.
    pub fn num_measurements(&self) -> usize {
        self.segments.last().map_or(0, |segment| {
            segment.measurement_start + segment.measurement_count
        })
    }
}

/// Linearize a [`Bloq`] into per-node Stim chunks (see [`BloqStimSegments`]),
/// noiseless and trusting the program.
///
/// Production callers want [`emit_bloq_stim`], which streams into one buffer.
/// This variant exists for per-node inspection and index alignment. Use
/// [`emit_bloq_stim_segments_with`] for noise or a checked input; noise is
/// already part of each node's instantiated circuit by the time it is chunked.
///
/// # Errors
///
/// Returns [`StimEmissionError`] if the program cannot be statically emitted.
pub fn emit_bloq_stim_segments(program: &Bloq) -> Result<BloqStimSegments, StimEmissionError> {
    emit_bloq_stim_segments_with(program, &BloqStimOptions::new())
}

/// [`emit_bloq_stim_segments`] under explicit [`BloqStimOptions`].
///
/// # Errors
///
/// Returns [`StimEmissionError`] for invalid, unsupported, or inconsistent input.
pub fn emit_bloq_stim_segments_with(
    program: &Bloq,
    options: &BloqStimOptions<'_>,
) -> Result<BloqStimSegments, StimEmissionError> {
    if options.align_moments {
        return Err(StimEmissionError::AlignMomentsWithSegments);
    }
    let (plans, order) = prepare_untransformed(program, options)?;
    BloqStimEmitter::new(program, options.noise, &plans).emit_order_segments(&order)
}

/// Emit the same program twice, clean and with `noise`, as `(clean, noisy)`.
///
/// A Monte-Carlo consumer needs both: the noisy circuit to sample and the
/// clean one to read reference signs and decoder recipes off. Because both
/// come from one call, this guarantees what a caller emitting twice can only
/// assert afterwards — the two share a node order and a measurement layout, so
/// a column resolved against one indexes the other. If they ever disagree the
/// emission is inconsistent and this fails rather than handing back a pair
/// whose indices silently diverge.
///
/// # Errors
///
/// Anything either emission would return on its own, plus
/// [`StimEmissionError::MalformedGraph`] if the two disagree on node order or
/// measurement layout.
pub fn emit_bloq_stim_segments_pair(
    program: &Bloq,
    noise: &NoiseModel,
) -> Result<(BloqStimSegments, BloqStimSegments), StimEmissionError> {
    reject_unsupported(program)?;
    let order = program.deterministic_emit_order()?;
    let trust = InputTrust::default();

    let clean_plans = emission_plans(program, None, trust)?;
    let clean = BloqStimEmitter::new(program, None, &clean_plans).emit_order_segments(&order)?;
    let noisy_plans = emission_plans(program, Some(noise), trust)?;
    let noisy =
        BloqStimEmitter::new(program, Some(noise), &noisy_plans).emit_order_segments(&order)?;

    let aligned = clean.segments.len() == noisy.segments.len()
        && std::iter::zip(&clean.segments, &noisy.segments).all(|(clean, noisy)| {
            clean.node_id == noisy.node_id
                && clean.measurement_start == noisy.measurement_start
                && clean.measurement_count == noisy.measurement_count
        });
    if !aligned {
        return Err(StimEmissionError::MalformedGraph(
            "clean and noisy emission disagree on node order or measurement layout",
        ));
    }
    Ok((clean, noisy))
}

struct BloqStimEmitter<'a> {
    program: &'a Bloq,
    level: &'a SubGraph,
    path: LevelPath,
    noise: Option<&'a NoiseModel>,
    attempt: Option<AttemptRender<'a>>,
    /// The per-node emission plans (IR-02). A multi-instance or noisy
    /// quantum node emits from its plan here instead of re-running the merge; the
    /// single-instance fast path emits its template verbatim and ignores its plan.
    plans: &'a ValidatedPlans,
    output: String,
    frame: MeasurementFrame,
    num_measurements: u32,
    layout: QubitLayout,
    indexed_coords: &'a [IVec2],
    resolved_loop_detector_states: ResolvedLoopStates,
    instance_measurements: FxHashMap<InstanceMeasurement, u32>,
    instance_templates: FxHashMap<TemplateInstanceId, TemplateId>,
    instance_loop_states: FxHashMap<InstanceLoopState, LoopStateId>,
    emitted_observables: FxHashSet<u32>,
    detector_metadata: Option<Vec<EmittedDetectorMetadata>>,
    /// Reused across nodes so per-node annotation vectors amortize capacity.
    annotations_scratch: StimCircuitAnnotations,
    /// Observable boundary logical operators grouped by their
    /// producing quantum node, then by observable index → (`operator_in`,
    /// `operator_out`). Emitted bracketing that node's circuit, so
    /// each Pauli face reads qubit state at the right temporal boundary.
    boundary_faces: FxHashMap<BloqNodeId, BTreeMap<u32, (PauliMap, PauliMap)>>,
}

struct AlignedNodeEmission {
    /// The already-instantiated circuit, with its entry ops temporarily removed.
    circuit: CoordCircuit,
    moments: Vec<Vec<Op>>,
    annotations: StimCircuitAnnotations,
    ends_with_tick: bool,
    boundary_faces: BTreeMap<u32, (PauliMap, PauliMap)>,
}

#[derive(Debug, Clone, Copy)]
enum AttemptMode {
    HonestT,
    SControl,
    Companion { cultivation: BloqNodeId },
}

#[derive(Debug, Clone, Copy)]
struct AttemptRender<'a> {
    mode: AttemptMode,
    noise: &'a NoiseModel,
}

impl AttemptRender<'_> {
    fn emits_observables(self) -> bool {
        matches!(self.mode, AttemptMode::Companion { .. })
    }

    fn enables_clifford_proxy(self) -> bool {
        !matches!(self.mode, AttemptMode::HonestT)
    }

    fn emits_postselection_tags(self) -> bool {
        matches!(self.mode, AttemptMode::Companion { .. })
    }

    fn noises_node(self, node: BloqNodeId) -> bool {
        !matches!(self.mode, AttemptMode::Companion { cultivation } if cultivation == node)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct EmittedDetectorMetadata {
    sign: bool,
    postselection: bool,
}

struct AttemptStimRender {
    text: String,
    measurement_columns: FxHashMap<InstanceMeasurement, u32>,
    detectors: Vec<EmittedDetectorMetadata>,
}

impl<'a> BloqStimEmitter<'a> {
    fn new(program: &'a Bloq, noise: Option<&'a NoiseModel>, plans: &'a ValidatedPlans) -> Self {
        Self::new_level(program, program.top(), LevelPath::default(), noise, plans)
    }

    fn new_level(
        program: &'a Bloq,
        level: &'a SubGraph,
        path: LevelPath,
        noise: Option<&'a NoiseModel>,
        plans: &'a ValidatedPlans,
    ) -> Self {
        Self {
            program,
            level,
            path,
            noise,
            attempt: None,
            plans,
            output: String::new(),
            frame: MeasurementFrame::default(),
            num_measurements: 0,
            layout: QubitLayout::default(),
            indexed_coords: &[],
            resolved_loop_detector_states: ResolvedLoopStates::default(),
            instance_measurements: FxHashMap::default(),
            instance_templates: FxHashMap::default(),
            instance_loop_states: FxHashMap::default(),
            emitted_observables: FxHashSet::default(),
            detector_metadata: None,
            annotations_scratch: StimCircuitAnnotations::default(),
            boundary_faces: FxHashMap::default(),
        }
    }

    fn new_attempt(
        program: &'a Bloq,
        level: &'a SubGraph,
        path: LevelPath,
        noise: &'a NoiseModel,
        mode: AttemptMode,
        plans: &'a ValidatedPlans,
    ) -> Self {
        let mut emitter = Self::new_level(program, level, path, None, plans);
        emitter.attempt = Some(AttemptRender { mode, noise });
        emitter.detector_metadata = Some(Vec::new());
        emitter
    }

    fn prepare(&mut self) -> Result<(), StimEmissionError> {
        self.instance_templates = self
            .program
            .levels()
            .flat_map(|(_, level)| {
                level.quantum_nodes().flat_map(|(_, quantum)| {
                    quantum
                        .instances
                        .iter()
                        .map(|instance| (instance.id, instance.template_id))
                })
            })
            .collect();
        self.build_global_layout()?;
        self.build_global_measurements()?;
        self.build_global_loop_states()?;
        self.build_boundary_faces()?;
        self.output
            .reserve(estimate_bloq_stim_capacity(self.program, self.layout.len()));
        self.emit_qubit_coords();
        Ok(())
    }

    fn emit_order_untransformed(
        mut self,
        order: &[BloqNodeId],
    ) -> Result<String, StimEmissionError> {
        self.prepare()?;
        for id in order {
            self.emit_node(*id, &self.level[*id])?;
        }
        Ok(self.output)
    }

    fn emit_order_aligned(mut self, order: &[BloqNodeId]) -> Result<String, StimEmissionError> {
        self.prepare()?;
        let (lanes, mut nodes) = self.prepare_aligned_nodes(order)?;

        for layer in align_moment_lanes(self.level, lanes)? {
            for slot in layer.slots {
                let mut completed = Vec::new();
                for entry in slot.entries {
                    let emission = nodes
                        .get_mut(&entry.node)
                        .expect("alignment returns a submitted node");
                    if self.emit_aligned_moment(entry.node, entry.moment, emission)? {
                        completed.push(entry.node);
                    }
                }
                self.output.push_str("TICK\n");
                let mut needs_boundary_tick = false;
                for &node_id in &completed {
                    let emission = nodes.get_mut(&node_id).expect("completed node is prepared");
                    if !emission.ends_with_tick {
                        continue;
                    }
                    needs_boundary_tick |=
                        self.emit_aligned_circuit(&emission.circuit, Some(&emission.annotations))?;
                }
                if needs_boundary_tick {
                    self.output.push_str("TICK\n");
                }
                for node_id in completed {
                    let emission = nodes.get(&node_id).expect("completed node is prepared");
                    for (index, (_, operator_out)) in &emission.boundary_faces {
                        self.emit_observable_pauli_line(*index, operator_out)?;
                    }
                }
            }
        }

        // Record sinks can move after all quantum measurements. `prepare`
        // already attached boundary operators to their quantum nodes.
        for &node_id in order {
            if self.level[node_id].try_quantum().is_none() {
                self.emit_node(node_id, &self.level[node_id])?;
            }
        }
        Ok(self.output)
    }

    fn prepare_aligned_nodes(
        &mut self,
        order: &[BloqNodeId],
    ) -> Result<(Vec<MomentLane>, FxHashMap<BloqNodeId, AlignedNodeEmission>), StimEmissionError>
    {
        let mut lanes = Vec::with_capacity(order.len());
        let mut nodes = FxHashMap::with_capacity_and_hasher(order.len(), Default::default());

        for &node_id in order {
            let node = &self.level[node_id];
            let Some(quantum) = node.try_quantum() else {
                continue;
            };
            let plan =
                self.plans
                    .get(&self.path, node_id)
                    .ok_or(StimEmissionError::MalformedGraph(
                        "quantum node has no emission plan",
                    ))?;
            let mut circuit = remapped_plan_circuit(&mut self.instance_measurements, plan);

            let mut annotations = StimCircuitAnnotations::default();
            node_annotations(
                &mut annotations,
                self.program,
                node,
                self.program.templates(),
                &self.instance_measurements,
                &self.instance_loop_states,
                NodeAnnotationOptions {
                    instance_templates: &self.instance_templates,
                    bodies: Some(&plan.bodies),
                    include_restarts: false,
                },
            )?;

            let qubits = circuit.qubits().into_iter().collect();
            let entry = circuit.entry_body();
            let body = circuit
                .body_mut(entry)
                .expect("entry body is created with the circuit");
            let ops = std::mem::take(body.ops_mut());
            let ends_with_tick = matches!(ops.last(), Some(Op::Tick));
            let segments = aligned_moment_segments(&ops)?;
            let (kinds, moments): (Vec<_>, Vec<_>) = segments
                .into_iter()
                .map(|segment| {
                    segment.map_or((None, Vec::new()), |segment| {
                        (Some(segment.kind), segment.ops)
                    })
                })
                .unzip();
            let boundary_faces = self.boundary_faces.remove(&node_id).unwrap_or_default();
            if moments.is_empty() && quantum.timeline.is_none() {
                let has_faces = boundary_faces
                    .values()
                    .any(|(input, output)| !input.is_empty() || !output.is_empty());
                if !annotations.detectors.is_empty()
                    || !annotations.repeat_states.is_empty()
                    || has_faces
                {
                    return Err(StimEmissionError::AlignMomentsWithEmptyMetadata);
                }
                continue;
            }
            lanes.push(MomentLane {
                node: node_id,
                moments: kinds,
                qubits,
            });
            nodes.insert(
                node_id,
                AlignedNodeEmission {
                    circuit,
                    moments,
                    annotations,
                    ends_with_tick,
                    boundary_faces,
                },
            );
        }
        Ok((lanes, nodes))
    }

    fn emit_aligned_moment(
        &mut self,
        node_id: BloqNodeId,
        moment: usize,
        emission: &mut AlignedNodeEmission,
    ) -> Result<bool, StimEmissionError> {
        let moment_count = emission.moments.len();
        let ops = emission
            .moments
            .get_mut(moment)
            .expect("alignment returns a submitted moment");
        let last = moment + 1 == moment_count;
        let finalize_now = last && !emission.ends_with_tick;

        if moment == 0 {
            let node = &self.level[node_id];
            self.emit_node_comment(node_id, node);
            for (index, (operator_in, _)) in &emission.boundary_faces {
                self.emit_observable_pauli_line(*index, operator_in)?;
            }
        }

        let entry = emission.circuit.entry_body();
        std::mem::swap(
            emission
                .circuit
                .body_mut(entry)
                .expect("entry body is created with the circuit")
                .ops_mut(),
            ops,
        );
        let result = self.emit_aligned_circuit(
            &emission.circuit,
            finalize_now.then_some(&emission.annotations),
        );
        std::mem::swap(
            emission
                .circuit
                .body_mut(entry)
                .expect("entry body is created with the circuit")
                .ops_mut(),
            ops,
        );
        result?;

        Ok(last)
    }

    fn emit_aligned_circuit(
        &mut self,
        circuit: &CoordCircuit,
        annotations: Option<&StimCircuitAnnotations>,
    ) -> Result<bool, CircuitError> {
        let mut mapper = PassthroughMeasurementMapper {
            num_measurements: self.num_measurements,
        };
        let mut ctx = StimEmitContext {
            layout: &self.layout,
            frame: &mut self.frame,
            resolved_loop_detector_states: &mut self.resolved_loop_detector_states,
            output: &mut self.output,
            indent: 0,
        };
        emit_stim_circuit(
            circuit,
            annotations,
            &mut ctx,
            &mut mapper,
            StimEmitOptions::default(),
        )
    }

    fn emit_order_attempt(
        mut self,
        order: &[BloqNodeId],
    ) -> Result<AttemptStimRender, StimEmissionError> {
        self.prepare()?;
        for id in order {
            self.emit_node(*id, &self.level[*id])?;
        }
        let mut measurement_columns = FxHashMap::with_capacity_and_hasher(
            self.instance_measurements.len(),
            Default::default(),
        );
        for (&measurement, &backend) in &self.instance_measurements {
            measurement_columns.insert(
                measurement,
                self.frame.resolve_absolute_measurement(backend)?,
            );
        }
        Ok(AttemptStimRender {
            text: self.output,
            measurement_columns,
            detectors: self
                .detector_metadata
                .take()
                .expect("attempt emitter owns detector metadata"),
        })
    }

    /// Linearize into per-node Stim chunks sharing one global measurement frame
    /// (see [`BloqStimSegments`]). Unlike [`Self::emit_order_untransformed`], the
    /// node text is drained after each node so callers can inspect and align one
    /// node in isolation. `prepare` emits only the `QUBIT_COORDS` preamble, taken as the
    /// header; the frame and global maps persist across the drain so cross-node
    /// `rec[-k]` detector targets still resolve once reassembled.
    fn emit_order_segments(
        mut self,
        order: &[BloqNodeId],
    ) -> Result<BloqStimSegments, StimEmissionError> {
        self.prepare()?;
        let header = std::mem::take(&mut self.output);
        let mut segments = Vec::with_capacity(order.len());
        for &node_id in order {
            let measurement_start = self.frame.emitted_count() as usize;
            self.emit_node(node_id, &self.level[node_id])?;
            segments.push(StimSegment {
                node_id,
                text: std::mem::take(&mut self.output),
                measurement_start,
                measurement_count: self.frame.emitted_count() as usize - measurement_start,
            });
        }
        Ok(BloqStimSegments { header, segments })
    }

    /// Emit one node at its position in the deterministic order.
    /// A quantum node instantiates its templates; an [`Observable`](ClassicalNode::Observable)
    /// emits its local records and records from its input fragments. Boundary
    /// operators are emitted at their quantum owners. Fragments and computations
    /// carry no standalone op; their records fold into their consuming observable. Regions and
    /// [`Discard`](ClassicalNode::Discard) are rejected by name when the walk reaches them.
    fn emit_node(&mut self, node_id: BloqNodeId, node: &BloqNode) -> Result<(), StimEmissionError> {
        // Prepend a provenance comment so a reader can trace each slice of the
        // circuit back to its source node. Many classical nodes emit nothing
        // (their contribution folds into an Observable elsewhere), so the
        // comment is rolled back if the node produced no text — no dangling
        // headers over empty regions.
        let comment_start = self.output.len();
        self.emit_node_comment(node_id, node);
        let after_comment = self.output.len();

        match (node.try_classical(), &node.kind) {
            (_, BloqNodeKind::Quantum(_)) => {
                // Bracket the node with its boundary logical operators (IR spec
                // §4.7): input faces read state entering the node (before its
                // circuit), output faces read on exit (after). Removed (not
                // borrowed) so the map entry is owned here — each node id appears
                // once in the emit order, so we never need it again.
                let faces = self
                    .attempt
                    .is_none_or(AttemptRender::emits_observables)
                    .then(|| self.boundary_faces.remove(&node_id))
                    .flatten();
                if let Some(faces) = &faces {
                    for (index, (operator_in, _)) in faces {
                        self.emit_observable_pauli_line(*index, operator_in)?;
                    }
                }
                self.emit_instantiated_node(node_id, node)?;
                if let Some(faces) = &faces {
                    for (index, (_, operator_out)) in faces {
                        self.emit_observable_pauli_line(*index, operator_out)?;
                    }
                }
            }
            (
                Some(ClassicalNode::Observable {
                    index: Some(index), ..
                }),
                _,
            ) => {
                if self.attempt.is_none_or(AttemptRender::emits_observables) {
                    self.emit_observable_node(*index, node_id)?;
                }
            }
            (
                Some(ClassicalNode::Observable { index: None, .. } | ClassicalNode::Compute { .. }),
                _,
            ) => {}
            // Every other kind is un-emittable (`Discard`, regions), rejected up
            // front by `reject_unsupported` — the single source of truth for
            // emittability — so the walk never reaches them.
            _ => {
                debug_assert!(
                    stim_unsupported_reason(node).is_some(),
                    "unhandled emittable node kind"
                );
                unreachable!("un-emittable node kind passed reject_unsupported")
            }
        }
        if self.output.len() == after_comment {
            self.output.truncate(comment_start);
        }
        Ok(())
    }

    /// Write a `# node <id>: <kind> <provenance>` header for the node about to
    /// be emitted. Rolled back by [`Self::emit_node`] if the node emits no text.
    fn emit_node_comment(&mut self, node_id: BloqNodeId, node: &BloqNode) {
        use std::fmt::Write as _;
        let kind: Cow<'static, str> = match (node.try_classical(), &node.kind) {
            (_, BloqNodeKind::Quantum(_)) => "quantum".into(),
            (
                Some(ClassicalNode::Observable {
                    index: Some(index), ..
                }),
                _,
            ) => format!("observable {index}").into(),
            (_, BloqNodeKind::Classical(_)) => "classical".into(),
            (_, BloqNodeKind::Region(_)) => "region".into(),
        };
        let _ = writeln!(
            self.output,
            "# node {}: {kind} {}",
            node_id.0, node.provenance
        );
    }

    /// Emit canonical record parity at the observable's position. Boundary
    /// operators stay at their quantum owners. Static Stim uses a zero decoder
    /// estimate, ignoring the symbolic decoder flips and affine reference sign.
    fn emit_observable_node(
        &mut self,
        index: u32,
        node_id: BloqNodeId,
    ) -> Result<(), StimEmissionError> {
        let value = self
            .level
            .resolve_classical(node_id, ClassicalAssignment::Uniform(false))?;
        self.emit_observable_records(index, &value.measurements)?;
        if self.emitted_observables.insert(index) {
            self.output.push_str("OBSERVABLE_INCLUDE(");
            push_int(&mut self.output, index);
            self.output.push_str(")\n");
        }
        Ok(())
    }

    /// Resolve an observable's named measurement sites to `rec[-k]` lookbacks
    /// against the global frame and emit them as one `OBSERVABLE_INCLUDE` line.
    /// A validated graph always names emitted measurements and reads them after
    /// their producing node (the read-after-measure `Order` edge); an unvalidated
    /// graph that breaks either invariant surfaces as [`MalformedGraph`](StimEmissionError::MalformedGraph).
    fn emit_observable_records(
        &mut self,
        index: u32,
        measurements: &BTreeSet<InstanceMeasurement>,
    ) -> Result<(), StimEmissionError> {
        let mut lookbacks = Vec::with_capacity(measurements.len());
        for measurement in measurements {
            let backend_id = *self.instance_measurements.get(measurement).ok_or(
                StimEmissionError::MalformedGraph(
                    "observable references an unemitted instance measurement",
                ),
            )?;
            lookbacks.push(self.frame.resolve_measurement(backend_id).map_err(
                |error| match error {
                    CircuitError::InvalidMeasurementId(_) => StimEmissionError::MalformedGraph(
                        "observable reads a measurement before its producing node",
                    ),
                    error => StimEmissionError::Circuit(error),
                },
            )?);
        }
        lookbacks.sort_unstable();
        crate::emit::emit_observable_include_records(&mut self.output, index, &lookbacks);
        if !lookbacks.is_empty() {
            self.emitted_observables.insert(index);
        }
        Ok(())
    }

    /// Group inline and fragment boundary operators by their quantum owner,
    /// combining faces per observable index for bracketed emission.
    fn build_boundary_faces(&mut self) -> Result<(), StimEmissionError> {
        use bloq_ir::BoundaryFace;

        let level = self.level;

        // Plain record observables need no instance→node boundary map.
        let has_operators = level.nodes().any(|(_, node)| {
            matches!(
                node.try_classical(),
                Some(ClassicalNode::Observable { operators, .. })
                    if !operators.is_empty()
            )
        });
        if !has_operators {
            return Ok(());
        }

        // Composition propagates owned boundaries by XOR path multiplicity.
        // Boolean value edges contribute parity only, never boundary operators.
        let mut consumers: FxHashMap<BloqNodeId, BTreeSet<u32>> = FxHashMap::default();
        for id in level.deterministic_emit_order()?.into_iter().rev() {
            let mut indices = BTreeSet::new();
            if let Some(ClassicalNode::Observable {
                index: Some(index), ..
            }) = level[id].try_classical()
            {
                indices.insert(*index);
            }
            for (consumer, _) in level.compose_consumers(id) {
                for index in consumers.get(&consumer).into_iter().flatten() {
                    if !indices.insert(*index) {
                        indices.remove(index);
                    }
                }
            }
            consumers.insert(id, indices);
        }

        // Each template instance is produced by exactly one quantum node.
        let mut instance_node: FxHashMap<TemplateInstanceId, BloqNodeId> = FxHashMap::default();
        for (id, quantum) in level.quantum_nodes() {
            for instance in &quantum.instances {
                instance_node.insert(instance.id, id);
            }
        }

        let mut faces: FxHashMap<BloqNodeId, BTreeMap<u32, (PauliMap, PauliMap)>> =
            FxHashMap::default();
        for (node, weight) in level.nodes() {
            let Some(ClassicalNode::Observable { operators, .. }) = weight.try_classical() else {
                continue;
            };
            let indices = &consumers[&node];
            if indices.is_empty() {
                continue;
            }
            // Each operator resolves its own producing node.
            for operator in operators {
                let producer = *instance_node.get(&operator.instance).ok_or(
                    StimEmissionError::MalformedGraph(
                        "observable boundary operator names an unproduced instance",
                    ),
                )?;
                for &index in indices {
                    let (operator_in, operator_out) =
                        faces.entry(producer).or_default().entry(index).or_default();
                    match operator.face {
                        BoundaryFace::Input => *operator_in = &*operator_in ^ &operator.operator,
                        BoundaryFace::Output => {
                            *operator_out = &*operator_out ^ &operator.operator;
                        }
                    }
                }
            }
        }
        self.boundary_faces = faces;
        Ok(())
    }

    /// Emit one boundary logical operator face as a positioned, Pauli-target
    /// `OBSERVABLE_INCLUDE`. An empty face (cancelled at a seam or
    /// absent) contributes nothing. A qubit missing from the layout means the
    /// graph was never validated, so it surfaces as [`MalformedGraph`](StimEmissionError::MalformedGraph) like the
    /// rest of this path.
    fn emit_observable_pauli_line(
        &mut self,
        index: u32,
        operator: &bloq_circuit::PauliMap,
    ) -> Result<(), StimEmissionError> {
        if operator.is_empty() {
            return Ok(());
        }
        let output_start = self.output.len();
        crate::emit::emit_observable_include_pauli_targets(
            &mut self.output,
            index,
            operator,
            &self.layout,
        )
        .map_err(|_| {
            StimEmissionError::MalformedGraph(
                "boundary operator names a qubit outside the global qubit layout",
            )
        })?;
        if self.output.len() != output_start {
            self.emitted_observables.insert(index);
        }
        Ok(())
    }

    fn build_global_layout(&mut self) -> Result<(), StimEmissionError> {
        let coords = self.program.sorted_layout_coords()?;

        self.layout = QubitLayout::new(coordinate_index(coords.iter().copied())?)?;
        self.indexed_coords = coords;
        Ok(())
    }

    /// Allocate backend measurement ids for every instance measurement.
    ///
    /// A dangling `template_id` is rejected here, up front — skipping it would
    /// defer the failure to a misleading lookup panic far from the defect —
    /// so the emit paths may treat every instance template as present.
    fn build_global_measurements(&mut self) -> Result<(), StimEmissionError> {
        // Measurement ids are allocated here from each quantum node's instances,
        // contiguously from 0. Classical / region nodes carry no instances, so
        // they are skipped here too — see the per-node emit walk in `emit_node`.
        let mut next_instance_measurement = 0u32;
        self.instance_measurements.clear();
        for (node_id, quantum) in self.level.quantum_nodes() {
            let templates = self
                .plans
                .get(&self.path, node_id)
                .map_or(self.program.templates(), |plan| {
                    plan.templates(self.program.templates())
                });
            for instance in &quantum.instances {
                let template = templates.get(instance.template_id).ok_or(
                    StimEmissionError::MalformedGraph("instance references a missing template"),
                )?;
                for record in template.circuit.meas_registry().records() {
                    self.instance_measurements.insert(
                        InstanceMeasurement {
                            instance: instance.id,
                            measurement: record.id,
                        },
                        next_instance_measurement,
                    );
                    next_instance_measurement = next_instance_measurement
                        .checked_add(1)
                        .expect("id space is u32; programs stay far below 2^32 measurements");
                }
            }
        }
        self.num_measurements = next_instance_measurement;
        // The frame's occurrence table is indexed by the same ids;
        // pre-sizing it avoids repeated reallocation during emission.
        self.frame
            .reserve_measurement_ids(next_instance_measurement as usize);
        Ok(())
    }

    /// Allocate global loop-state ids. Rejects dangling `template_id`s for the
    /// same reason as [`Self::build_global_measurements`].
    fn build_global_loop_states(&mut self) -> Result<(), StimEmissionError> {
        self.instance_loop_states.clear();
        let mut next_state = 0;
        for (node_id, quantum) in self.level.quantum_nodes() {
            let templates = self
                .plans
                .get(&self.path, node_id)
                .map_or(self.program.templates(), |plan| {
                    plan.templates(self.program.templates())
                });
            for instance in &quantum.instances {
                let template = templates.get(instance.template_id).ok_or(
                    StimEmissionError::MalformedGraph("instance references a missing template"),
                )?;
                for state in &template.repeat_states {
                    let key = InstanceLoopState {
                        instance: instance.id,
                        state: state.state,
                    };
                    if self.instance_loop_states.contains_key(&key) {
                        continue;
                    }
                    self.instance_loop_states
                        .insert(key, LoopStateId(next_state));
                    next_state = next_state
                        .checked_add(1)
                        .expect("id space is u32; programs stay far below 2^32 loop states");
                }
            }
        }
        Ok(())
    }

    fn emit_qubit_coords(&mut self) {
        // `indexed_coords` is the sorted global layout, so a slice position is
        // its backend qubit index and the pairs are already in index order.
        // Copy the shared slice out first so the borrow is independent of the
        // `&mut self.output` the helper writes into.
        let coords = self.indexed_coords;
        crate::emit::emit_qubit_coords_lines(
            &mut self.output,
            coords
                .iter()
                .enumerate()
                .map(|(index, coord)| (*coord, index as u32)),
        );
    }

    fn collect_detector_metadata(&mut self) -> Result<(), StimEmissionError> {
        let Some(metadata) = &mut self.detector_metadata else {
            return Ok(());
        };
        if !self.annotations_scratch.repeat_states.is_empty()
            || self
                .annotations_scratch
                .detectors
                .iter()
                .any(|detector| detector.scope != StimAnnotationScope::TopLevel)
        {
            return Err(StimEmissionError::MalformedGraph(
                "isolated-T attempt must be flattened before detector indexing",
            ));
        }
        metadata.extend(self.annotations_scratch.detectors.iter().map(|detector| {
            EmittedDetectorMetadata {
                sign: detector.parity.sign(),
                postselection: detector.postselection,
            }
        }));
        Ok(())
    }

    fn emit_instantiated_node(
        &mut self,
        node_id: BloqNodeId,
        node: &BloqNode,
    ) -> Result<(), StimEmissionError> {
        // Single instances emit their template circuit verbatim: there is nothing
        // to merge, and routing them through the merge would reshape the output
        // (e.g. inserting ticks around REPEAT blocks the template never had).
        if node.expect_quantum().instances.len() == 1 && self.noise.is_none() {
            return self.emit_single_instance_node(node_id, node);
        }
        let instantiated =
            self.plans
                .get(&self.path, node_id)
                .ok_or(StimEmissionError::MalformedGraph(
                    "quantum node has no emission plan",
                ))?;
        let circuit = remapped_plan_circuit(&mut self.instance_measurements, instantiated);
        node_annotations(
            &mut self.annotations_scratch,
            self.program,
            node,
            instantiated.templates(self.program.templates()),
            &self.instance_measurements,
            &self.instance_loop_states,
            NodeAnnotationOptions {
                instance_templates: &self.instance_templates,
                bodies: Some(&instantiated.bodies),
                include_restarts: self.attempt.is_some(),
            },
        )?;
        let mut mapper = PassthroughMeasurementMapper {
            num_measurements: self.num_measurements,
        };
        let mut ctx = StimEmitContext {
            layout: &self.layout,
            frame: &mut self.frame,
            resolved_loop_detector_states: &mut self.resolved_loop_detector_states,
            output: &mut self.output,
            indent: 0,
        };
        let needs_boundary_tick = emit_stim_circuit(
            &circuit,
            Some(&self.annotations_scratch),
            &mut ctx,
            &mut mapper,
            StimEmitOptions::default(),
        )?;
        self.collect_detector_metadata()?;
        if needs_boundary_tick {
            self.output.push_str("TICK\n");
        }
        Ok(())
    }

    fn emit_single_instance_node(
        &mut self,
        node_id: BloqNodeId,
        node: &BloqNode,
    ) -> Result<(), StimEmissionError> {
        let instance = node
            .expect_quantum()
            .instances
            .first()
            .expect("single-instance path has one instance");
        let template = self
            .program
            .templates()
            .get(instance.template_id)
            .expect("prepare rejected instances referencing missing templates");
        // A single instance emits its template bodies verbatim, so every body
        // maps to itself; the identity mapping skips building (and hashing) a
        // per-node body map.
        node_annotations(
            &mut self.annotations_scratch,
            self.program,
            node,
            self.program.templates(),
            &self.instance_measurements,
            &self.instance_loop_states,
            NodeAnnotationOptions {
                instance_templates: &self.instance_templates,
                bodies: None,
                include_restarts: self.attempt.is_some(),
            },
        )?;
        let noisy_circuit = self
            .attempt
            .filter(|attempt| attempt.noises_node(node_id))
            .map(|attempt| {
                let ideal_qubits = if instance.provenance.is_spatial_port_substitution() {
                    template.qubits()
                } else {
                    &[]
                };
                attempt
                    .noise
                    .noisy_circuit_excluding(&template.circuit, ideal_qubits)
            })
            .transpose()?;
        let circuit = noisy_circuit.as_ref().unwrap_or(&template.circuit);
        let enable_clifford_proxy = self
            .attempt
            .is_some_and(AttemptRender::enables_clifford_proxy);
        let allow_non_clifford = matches!(
            self.attempt.map(|attempt| attempt.mode),
            Some(AttemptMode::HonestT)
        );
        let emit_postselection_tags = self
            .attempt
            .is_none_or(AttemptRender::emits_postselection_tags);
        let mut mapper = SingleInstanceMeasurementMapper {
            instance: instance.id,
            measurements: &self.instance_measurements,
            num_measurements: self.num_measurements,
        };
        let mut ctx = StimEmitContext {
            layout: &self.layout,
            frame: &mut self.frame,
            resolved_loop_detector_states: &mut self.resolved_loop_detector_states,
            output: &mut self.output,
            indent: 0,
        };
        let needs_boundary_tick = emit_stim_circuit(
            circuit,
            Some(&self.annotations_scratch),
            &mut ctx,
            &mut mapper,
            StimEmitOptions {
                enable_clifford_proxy,
                allow_non_clifford,
                emit_postselection_tags,
                qubit_offset: instance.offset,
            },
        )?;
        self.collect_detector_metadata()?;
        if needs_boundary_tick {
            self.output.push_str("TICK\n");
        }
        Ok(())
    }
}

/// The first gate the static backend cannot emit, in level then node order, or
/// `None`.
///
/// Gate preflight recurses through every level (region bodies included) via
/// [`Bloq::levels`], because a non-Clifford gate can hide in a region body's
/// template. Node preflight ([`first_unsupported_program_node`]) scans only the
/// top level: any region there is rejected by name before its body is ever
/// inspected, so it never needs to descend.
///
/// A node carries many instances of one template, so each template circuit is
/// scanned at most once (memoized per [`TemplateId`]).
fn first_unsupported_program_gate(program: &Bloq) -> Option<GateType> {
    let templates = program.templates();
    let mut scanned: FxHashMap<TemplateId, Option<GateType>> = FxHashMap::default();
    for (_, level) in program.levels() {
        for (_, quantum) in level.quantum_nodes() {
            for instance in &quantum.instances {
                let unsupported = *scanned.entry(instance.template_id).or_insert_with(|| {
                    templates
                        .get(instance.template_id)
                        .and_then(|template| first_unsupported_gate(&template.circuit))
                });
                if unsupported.is_some() {
                    return unsupported;
                }
            }
        }
    }
    None
}

/// Flat per-instruction output budget for the reserve estimate.
///
/// Walking every op and its qubit list to compute an exact reserve costs a
/// measurable fraction of emission itself, so the reserve uses op counts
/// only. Suite-wide averages sit near 90 bytes per op (dominated by gate
/// qubit lists and detector lookbacks); 128 leaves headroom so reallocation
/// stays rare at the cost of a modest over-reserve.
const STIM_BYTES_PER_OP: usize = 128;

/// Clone a merged plan into the emitter-wide measurement namespace.
fn remapped_plan_circuit(
    instance_measurements: &mut FxHashMap<InstanceMeasurement, u32>,
    plan: &NodeEmissionPlan,
) -> CoordCircuit {
    let mut circuit = plan.circuit.clone();
    let mut local_measurements =
        FxHashMap::with_capacity_and_hasher(plan.measurements.len(), Default::default());
    // Each merged id uses the minimum preallocated id of its folded sources.
    for (local, sources) in plan.grouped_measurements() {
        let backend = sources
            .iter()
            .map(|source| {
                instance_measurements.get(source).copied().expect(
                    "build_global_measurements allocated an id for every instance measurement",
                )
            })
            .min()
            .expect("a grouped-measurement bucket is never empty");
        for source in sources {
            instance_measurements.insert(source, backend);
        }
        local_measurements.insert(local, backend);
    }
    circuit.remap_measurement_ids(&local_measurements);
    circuit
}

fn estimate_bloq_stim_capacity(program: &Bloq, global_qubits: usize) -> usize {
    let qubit_coords = global_qubits * 24;
    let templates = program.templates();
    let node_bodies = program
        .quantum_nodes()
        .map(|(_, quantum)| {
            // A node's emitted text comes from each instance's template circuit,
            // so the reserve sums those; the node itself carries no ops. Classical
            // / region nodes emit nothing and so are skipped (filtered above).
            let body = quantum
                .instances
                .iter()
                .filter_map(|instance| templates.get(instance.template_id))
                .map(|template| estimate_circuit_stim_capacity(&template.circuit))
                .sum::<usize>();
            body + 8
        })
        .sum::<usize>();
    qubit_coords + node_bodies
}

/// Estimate the emitted size of a node circuit from its op counts alone.
///
/// Repeat bodies live in the same circuit, so counting every body covers the
/// nesting without walking individual ops.
fn estimate_circuit_stim_capacity(circuit: &bloq_circuit::CoordCircuit) -> usize {
    (0..circuit.body_count())
        .map(|index| {
            circuit
                .body(BodyId(index as u32))
                .map_or(0, |body| body.ops().len())
        })
        .sum::<usize>()
        * STIM_BYTES_PER_OP
}

/// A measurement id passes through unchanged when it is live: global ids are
/// allocated contiguously from 0, so any id below the total count exists.
/// Otherwise it refers to a measurement outside the emitted range and is rejected.
fn passthrough_live_measurement(
    measurement: u32,
    num_measurements: u32,
) -> Result<u32, bloq_circuit::CircuitError> {
    if measurement < num_measurements {
        return Ok(measurement);
    }
    Err(bloq_circuit::CircuitError::InvalidMeasurementId(
        measurement,
    ))
}

/// Passes already-global measurement ids through unchanged, rejecting any id
/// not live in this node. Used by the multi-instance merge path, whose merged
/// circuit already carries backend-global ids.
struct PassthroughMeasurementMapper {
    num_measurements: u32,
}

impl StimMeasurementMapper for PassthroughMeasurementMapper {
    fn map_measurement(&mut self, measurement: u32) -> Result<u32, bloq_circuit::CircuitError> {
        passthrough_live_measurement(measurement, self.num_measurements)
    }
}

struct SingleInstanceMeasurementMapper<'a> {
    instance: TemplateInstanceId,
    measurements: &'a FxHashMap<InstanceMeasurement, u32>,
    num_measurements: u32,
}

impl StimMeasurementMapper for SingleInstanceMeasurementMapper<'_> {
    fn map_measurement(&mut self, measurement: u32) -> Result<u32, bloq_circuit::CircuitError> {
        let measurement = InstanceMeasurement {
            instance: self.instance,
            measurement,
        };
        self.measurements.get(&measurement).copied().ok_or(
            bloq_circuit::CircuitError::InvalidMeasurementId(measurement.measurement),
        )
    }

    fn map_annotation_measurement(
        &mut self,
        measurement: u32,
    ) -> Result<u32, bloq_circuit::CircuitError> {
        passthrough_live_measurement(measurement, self.num_measurements)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct InstanceLoopState {
    instance: TemplateInstanceId,
    state: LoopStateId,
}

/// Rebuilds `out`'s side-table annotations for `node`, reusing its buffers.
///
/// One reusable [`StimCircuitAnnotations`] is threaded across every node so
/// the per-node detector and repeat-state vectors amortize their capacity.
struct NodeAnnotationOptions<'a> {
    instance_templates: &'a FxHashMap<TemplateInstanceId, TemplateId>,
    bodies: Option<&'a FxHashMap<(TemplateInstanceId, BodyId), BodyId>>,
    include_restarts: bool,
}

fn node_annotations(
    out: &mut StimCircuitAnnotations,
    program: &Bloq,
    node: &BloqNode,
    templates: &bloq_ir::lowering::BloqTemplatePool,
    measurements: &FxHashMap<InstanceMeasurement, u32>,
    loop_states: &FxHashMap<InstanceLoopState, LoopStateId>,
    options: NodeAnnotationOptions<'_>,
) -> Result<(), StimEmissionError> {
    let NodeAnnotationOptions {
        instance_templates,
        bodies,
        include_restarts,
    } = options;
    let map_body = |instance, body| {
        bodies.map_or(body, |bodies| {
            bodies
                .get(&(instance, body))
                .copied()
                .expect("the emission plan maps every body of every merged instance")
        })
    };
    let StimCircuitAnnotations {
        detectors,
        repeat_states,
    } = out;
    detectors.clear();
    repeat_states.clear();
    let node = node.expect_quantum();
    program.check_detector_bundle_bindings(node, |instance| {
        instance_templates.get(&instance).copied()
    })?;
    for instance in &node.instances {
        let template = templates
            .get(instance.template_id)
            .expect("prepare rejected instances referencing missing templates");
        for detector in &template.detectors {
            let scope = match detector.scope {
                TemplateDetectorScope::TopLevel => StimAnnotationScope::TopLevel,
                TemplateDetectorScope::RepeatBody { body } => StimAnnotationScope::RepeatBody {
                    body: map_body(instance.id, body),
                },
            };
            detectors.push(StimDetector {
                scope,
                parity: remap_template_parity(
                    &detector.parity,
                    instance.id,
                    measurements,
                    loop_states,
                ),
                coords: detector
                    .coords
                    .as_ref()
                    .map(|coords| bloq_circuit::translate_detector_coords(coords, instance.offset)),
                postselection: false,
            });
        }
        if include_restarts {
            detectors.extend(template.restarts.iter().map(|restart| StimDetector {
                scope: StimAnnotationScope::TopLevel,
                parity: remap_template_parity(
                    &restart.parity,
                    instance.id,
                    measurements,
                    loop_states,
                ),
                coords: None,
                postselection: true,
            }));
        }
        for state in &template.repeat_states {
            let state_id = InstanceLoopState {
                instance: instance.id,
                state: state.state,
            };
            repeat_states.push(StimRepeatState {
                body: map_body(instance.id, state.body),
                state: loop_states
                    .get(&state_id)
                    .copied()
                    .expect("build_global_loop_states allocated an id for every instance state"),
                initial: remap_template_parity(
                    &state.initial,
                    instance.id,
                    measurements,
                    loop_states,
                ),
                next: remap_template_parity(&state.next, instance.id, measurements, loop_states),
            });
        }
    }
    for detector in program.node_detectors(node)? {
        detectors.push(StimDetector {
            scope: StimAnnotationScope::TopLevel,
            parity: remap_instance_parity(&detector.parity(), measurements)?,
            coords: detector.coords().map(Iterator::collect),
            postselection: false,
        });
    }
    if include_restarts {
        for restart in &node.restarts {
            detectors.push(StimDetector {
                scope: StimAnnotationScope::TopLevel,
                parity: remap_instance_parity(&restart.parity, measurements)?,
                coords: None,
                postselection: true,
            });
        }
    }
    repeat_states.sort_by_key(|state| (state.body, state.state));
    Ok(())
}

fn remap_template_parity(
    parity: &bloq_ir::lowering::TemplateDetectorParity,
    instance: bloq_ir::lowering::TemplateInstanceId,
    measurements: &FxHashMap<InstanceMeasurement, u32>,
    loop_states: &FxHashMap<InstanceLoopState, LoopStateId>,
) -> bloq_circuit::DetectorParity {
    // `from_terms` collects the iterator into the parity's own (inline-capable)
    // term buffer, so feed it the mapped terms directly rather than through an
    // intermediate heap `Vec`.
    bloq_circuit::DetectorParity::from_terms(parity.terms().iter().copied().map(
        |term| match term {
            bloq_circuit::DetectorTerm::Measurement(measurement) => {
                let measurement = InstanceMeasurement {
                    instance,
                    measurement,
                };
                measurements
                    .get(&measurement)
                    .copied()
                    .map(bloq_circuit::DetectorTerm::Measurement)
                    .expect(
                        "build_global_measurements allocated an id for every instance measurement",
                    )
            }
            bloq_circuit::DetectorTerm::LoopState(state) => {
                let state = InstanceLoopState { instance, state };
                loop_states
                    .get(&state)
                    .copied()
                    .map(bloq_circuit::DetectorTerm::LoopState)
                    .expect("build_global_loop_states allocated an id for every instance state")
            }
        },
    ))
    .with_sign(parity.sign())
}

fn remap_instance_parity(
    parity: &bloq_ir::NodeDetectorParity,
    measurements: &FxHashMap<InstanceMeasurement, u32>,
) -> Result<bloq_circuit::DetectorParity, StimEmissionError> {
    if let Some(state) = parity.terms().iter().find_map(|term| match term {
        bloq_circuit::DetectorTerm::LoopState(state) => Some(*state),
        bloq_circuit::DetectorTerm::Measurement(_) => None,
    }) {
        return Err(StimEmissionError::NodeLoopStateUnsupported(state));
    }
    parity.try_map_measurements(|measurement| {
        measurements
            .get(&measurement)
            .copied()
            .ok_or(StimEmissionError::UnknownInstanceMeasurement(measurement))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use bloq_circuit::{CircuitBody, DetectorParity, DetectorTerm, PauliBasis};
    use bloq_ir::{
        BloqEdge, BoundaryFace, ClassicalExpr, MemoryRoundTarget, NodeDetector, QuantumTimeline,
        SourceBlockRef, TemplateDetector,
        lowering::{BloqTemplate, InstanceBoundaryOperator, TemplateInstance, TemplateRepeatState},
    };
    use glam::{ivec2, ivec3};

    fn compiled_isolated_t(distance: u32, latency: u32) -> Bloq {
        let graph = bloq_graph::parse_blog_to_graph(
            "BLOG 1.0\n\n  0: T [0, 0, 0]\n  1: Port [0, 0, 1]\n  [0, 0, 0] -> +Z\n",
        )
        .expect("isolated T BLOG parses");
        let mut program =
            bloq_compile::CompileContext::new(bloq_compile::CompileConfig::new(distance))
                .compile(&graph)
                .expect("isolated T compiles")
                .bloq;
        if latency != 0 {
            let path = isolated_rus_path(&program).expect("RUS path");
            let terminal = program
                .level_at(&path)
                .expect("RUS body")
                .deterministic_emit_order()
                .expect("body order")
                .into_iter()
                .rev()
                .find(|&node| {
                    program
                        .level_at(&path)
                        .expect("RUS body")
                        .node(node)
                        .is_some_and(|node| node.try_quantum().is_some())
                })
                .expect("escape stage");
            program
                .insert_memory_rounds(
                    MemoryRoundTarget::After {
                        path,
                        node: terminal,
                    },
                    latency,
                )
                .expect("append decoder latency");
        }
        program
    }

    #[test]
    fn isolated_t_frame_inversions_preserve_columns_and_change_probe_corrections() {
        let program = compiled_isolated_t(3, 0);
        let baseline = emit_isolated_t_attempts(&program, 0.0).unwrap();
        let probe_signs = |manifest: &IsolatedTAttemptManifest| {
            let x = manifest.frame_x.sign;
            let z = manifest.frame_z.sign;
            std::array::from_fn::<_, 3, _>(|axis| {
                manifest.exp_val_signs[axis] * if [z, x ^ z, x][axis] { -1 } else { 1 }
            })
        };
        for invert_x in [true, false] {
            let mut changed = program.clone();
            let frames = changed.output_frames();
            let frame = if invert_x { frames[0].x } else { frames[0].z };
            let Some(ClassicalNode::Compute { expr }) =
                changed.node_mut(frame).unwrap().try_classical_mut()
            else {
                panic!("output frame is a Compute");
            };
            *expr = ClassicalExpr::Not(Box::new(expr.clone()));
            changed.validate().unwrap();
            let artifacts = emit_isolated_t_attempts(&changed, 0.0).unwrap();
            let mut expected = baseline.clone();
            if invert_x {
                expected.manifest.frame_x.sign ^= true;
            } else {
                expected.manifest.frame_z.sign ^= true;
            }
            assert_eq!(artifacts, expected);
            let before = probe_signs(&baseline.manifest);
            let after = probe_signs(&artifacts.manifest);
            let multipliers = if invert_x { [1, -1, -1] } else { [-1, -1, 1] };
            assert_eq!(
                after,
                std::array::from_fn(|axis| before[axis] * multipliers[axis])
            );
        }
    }

    #[test]
    fn isolated_t_attempt_emits_aligned_physical_and_causal_circuits() {
        let artifacts = emit_isolated_t_attempts(&compiled_isolated_t(3, 2), 1e-3)
            .expect("emit isolated T attempt");
        let detector_count = |text: &str| {
            text.lines()
                .filter(|line| line.starts_with("DETECTOR"))
                .count()
        };
        let exp_count = |text: &str| {
            text.lines()
                .filter(|line| line.starts_with("EXP_VAL "))
                .count()
        };

        assert!(artifacts.physical_t.contains("T_DAG"));
        assert!(!artifacts.physical_s.contains("T_DAG"));
        assert!(artifacts.physical_s.contains("S_DAG"));
        assert_eq!(exp_count(&artifacts.physical_s), 3);
        assert_eq!(exp_count(&artifacts.physical_t), 3);
        assert_eq!(exp_count(&artifacts.companion), 0);
        assert!(!artifacts.physical_s.contains("OBSERVABLE_INCLUDE"));
        assert!(!artifacts.physical_t.contains("OBSERVABLE_INCLUDE"));
        assert!(artifacts.companion.contains("OBSERVABLE_INCLUDE"));
        let noise_count = |text: &str| {
            text.lines()
                .filter(|line| line.starts_with("DEPOLARIZE") || line.contains("_ERROR("))
                .count()
        };
        assert_eq!(
            noise_count(&artifacts.physical_s),
            noise_count(&artifacts.physical_t)
        );
        assert!(noise_count(&artifacts.physical_s) > noise_count(&artifacts.companion));
        assert!(noise_count(&artifacts.companion) > 0);

        let expected_detectors = artifacts.manifest.detector_signs.len();
        assert_eq!(detector_count(&artifacts.physical_s), expected_detectors);
        assert_eq!(detector_count(&artifacts.physical_t), expected_detectors);
        assert_eq!(detector_count(&artifacts.companion), expected_detectors);
        let sheeted = format!("{}{}", artifacts.companion, artifacts.sheets);
        assert_eq!(detector_count(&sheeted), expected_detectors);
        assert_eq!(
            artifacts
                .companion
                .lines()
                .filter(|line| line.starts_with("DETECTOR[POST-SELECTION]"))
                .count(),
            artifacts.manifest.postselection_detectors.len()
        );
        assert!(!artifacts.physical_s.contains("DETECTOR[POST-SELECTION]"));
        assert!(!artifacts.physical_t.contains("DETECTOR[POST-SELECTION]"));
        assert_eq!(
            sheeted
                .lines()
                .filter(|line| line.starts_with("DETECTOR[POST-SELECTION]"))
                .count(),
            artifacts.manifest.postselection_detectors.len()
        );
        // The split is only sound if reassembly is exact: the suffix must add
        // observables and nothing else.
        assert!(
            artifacts
                .sheets
                .lines()
                .all(|line| line.starts_with("OBSERVABLE_INCLUDE(")),
            "sheets suffix carries only observables: {}",
            artifacts.sheets
        );
        assert!(!artifacts.manifest.postselection_detectors.is_empty());
        assert_eq!(artifacts.manifest.frontier_sheets.len(), 8);
        assert_eq!(artifacts.manifest.exp_val_signs, [1, 1, 1]);
        assert_eq!(
            artifacts.manifest.frame_x.gap_observable,
            artifacts.manifest.gap_z_observable
        );
        assert_eq!(
            artifacts.manifest.frame_z.gap_observable,
            artifacts.manifest.gap_x_observable
        );
    }

    #[test]
    fn isolated_t_attempt_matches_d5_m10_causal_cut() {
        let artifacts = emit_isolated_t_attempts(&compiled_isolated_t(5, 10), 1e-3)
            .expect("emit d5/m10 isolated T attempt");
        let measurement_count = artifacts
            .physical_t
            .lines()
            .filter_map(|line| {
                let mut words = line.split_whitespace();
                let gate = words.next()?.split('(').next()?;
                matches!(gate, "M" | "MX" | "MY" | "MPP").then(|| words.count())
            })
            .sum::<usize>();
        let detector_count = artifacts
            .physical_t
            .lines()
            .filter(|line| line.starts_with("DETECTOR"))
            .count();

        assert_eq!(measurement_count, 448);
        // 35 post-selection checks: the six first-round/temporal surface-band
        // parities are ordinary decoder detectors, not restart conditions.
        assert_eq!(detector_count, 424);
        assert_eq!(artifacts.manifest.detector_signs.len(), 424);
        assert_eq!(artifacts.manifest.postselection_detectors.len(), 35);
        assert_eq!(
            artifacts
                .companion
                .lines()
                .filter(|line| line.starts_with("DETECTOR[POST-SELECTION]"))
                .count(),
            35
        );
        assert!(!artifacts.physical_s.contains("DETECTOR[POST-SELECTION]"));
        assert!(!artifacts.physical_t.contains("DETECTOR[POST-SELECTION]"));
        assert_eq!(artifacts.manifest.frontier_sheets.len(), 24);
        assert_ne!(
            artifacts.manifest.gap_z_observable,
            artifacts.manifest.gap_x_observable
        );
        assert_eq!(
            artifacts.manifest.frame_x.measurements,
            [46, 95, 97, 112, 114, 116, 118, 120, 122]
        );
        assert_eq!(
            artifacts.manifest.frame_z.measurements,
            [15, 17, 18, 124, 125, 126, 127, 128, 129]
        );
        assert!(
            artifacts
                .manifest
                .frontier_sheets
                .iter()
                .all(|sheet| !sheet.sign)
        );
    }

    #[test]
    fn isolated_t_attempt_rejects_invalid_noise_probability() {
        let program = Bloq::new();
        for probability in [f64::NAN, f64::INFINITY, -0.1, 1.1] {
            assert_eq!(
                emit_isolated_t_attempts(&program, probability),
                Err(StimEmissionError::InvalidNoiseProbability)
            );
        }
    }

    #[test]
    fn signed_logical_y_keeps_global_phase() {
        let coord = ivec2(0, 0);
        let x = PauliMap::from_unique_entries([(coord, Pauli::Z)]);
        let z = PauliMap::from_unique_entries([(coord, Pauli::X)]);
        let (y, sign) = signed_logical_y(&x, &z).expect("iZX is Hermitian");

        assert_eq!(y.get(&coord), Some(&Pauli::Y));
        assert_eq!(sign, -1);
    }

    #[test]
    fn emit_bloq_stim_rejects_repeat_until_success_by_name() {
        // A `RepeatUntilSuccess` is postselected retry the static backend cannot
        // emit, so it is rejected by name.
        let mut bloq = Bloq::new();
        let body = SubGraph::new();
        bloq.add_node(BloqNode::region(RegionNode::RepeatUntilSuccess {
            body,
            restart_condition: ClassicalExpr::Const(false),
            restart_source: None,
        }));
        assert_eq!(
            emit_bloq_stim(&bloq),
            Err(StimEmissionError::UnsupportedNode(
                "RepeatUntilSuccess (postselected retry)"
            ))
        );
    }

    #[test]
    fn emit_bloq_stim_folds_observable_outputs_but_rejects_activation() {
        // A complete observable's static flip estimate is zero. Consumers of
        // its corrected output or structural composition therefore receive the
        // same measurement recipe; its flip output contributes no records.
        let mut graph = Bloq::new();
        let node = measured_program_node(
            &mut graph,
            ivec3(0, 0, 0),
            ivec2(0, 0),
            TemplateInstanceId(0),
        );
        let quantum = graph.add_node(node);
        let read_observable = graph.add_node(BloqNode::classical(ClassicalNode::Observable {
            index: Some(0),
            measurements: vec![InstanceMeasurement {
                instance: TemplateInstanceId(0),
                measurement: 0,
            }],
            operators: vec![],
        }));
        graph.add_edge(quantum, read_observable, BloqEdge::Order);
        let corrected_read = graph.add_node(BloqNode::classical(ClassicalNode::observable(1)));
        graph.add_edge(read_observable, corrected_read, BloqEdge::value(0));
        let composed = graph.add_node(BloqNode::classical(ClassicalNode::observable(2)));
        graph.add_edge(read_observable, composed, BloqEdge::compose(0));
        let flipped = graph.add_node(BloqNode::classical(ClassicalNode::observable(3)));
        graph.add_edge(read_observable, flipped, BloqEdge::flip(0));

        let text = emit_bloq_stim(&graph).expect("observable outputs reduce to records");
        for index in [0, 1, 2] {
            assert!(
                text.contains(&format!("OBSERVABLE_INCLUDE({index}) rec[-1]")),
                "observable output should fold to its physical record: {text}"
            );
        }

        assert!(text.contains("OBSERVABLE_INCLUDE(3)\n"), "{text}");
        assert!(!text.contains("OBSERVABLE_INCLUDE(3) rec"), "{text}");

        graph.node_mut(corrected_read).unwrap().activation = Some(1);
        graph.add_edge(read_observable, corrected_read, BloqEdge::value(1));
        assert!(!graph.has_conditional_membership());
        assert_eq!(
            emit_bloq_stim(&graph),
            Err(StimEmissionError::UnsupportedNode(
                "activation (conditional execution)"
            ))
        );
    }

    #[test]
    fn complete_observable_emits_inline_records_and_owned_boundary_faces() {
        let qubit = ivec2(0, 0);
        let instance = TemplateInstanceId(0);
        let mut graph = Bloq::new();
        let quantum = measured_program_node(&mut graph, ivec3(0, 0, 0), qubit, instance);
        let quantum = graph.add_node(quantum);
        let operator = [(qubit, Pauli::Z)].into_iter().collect::<PauliMap>();
        let observable = graph.add_node(BloqNode::classical(ClassicalNode::Observable {
            index: Some(3),
            measurements: vec![
                InstanceMeasurement {
                    instance,
                    measurement: 0
                };
                3
            ],
            operators: vec![
                InstanceBoundaryOperator {
                    instance,
                    face: BoundaryFace::Input,
                    operator: operator.clone(),
                },
                InstanceBoundaryOperator {
                    instance,
                    face: BoundaryFace::Output,
                    operator: operator.clone(),
                },
            ],
        }));
        graph.add_edge(quantum, observable, BloqEdge::Order);

        assert_eq!(
            observable_boundary_basis(graph.top(), observable),
            Ok(Basis::Z)
        );
        for aligned in [false, true] {
            let text =
                emit_bloq_stim_with(&graph, &BloqStimOptions::new().with_align_moments(aligned))
                    .unwrap();
            let input = text.find("OBSERVABLE_INCLUDE(3) Z0").unwrap();
            let measurement = text.find("M 0").unwrap();
            let output = text.rfind("OBSERVABLE_INCLUDE(3) Z0").unwrap();
            assert!(input < measurement && measurement < output, "{text}");
            assert!(text.contains("OBSERVABLE_INCLUDE(3) rec[-1]"), "{text}");
        }

        // A shared fragment and inline binding at the same owned face cancel.
        let include = graph.add_node(BloqNode::classical(ClassicalNode::observable_fragment(
            vec![],
            vec![InstanceBoundaryOperator {
                instance,
                face: BoundaryFace::Output,
                operator,
            }],
        )));
        graph.add_edge(include, observable, BloqEdge::compose(0));
        let text = emit_bloq_stim(&graph).unwrap();
        assert_eq!(
            text.matches("OBSERVABLE_INCLUDE(3) Z0").count(),
            1,
            "{text}"
        );
        assert!(text.find("OBSERVABLE_INCLUDE(3) Z0").unwrap() < text.find("M 0").unwrap());
    }

    #[test]
    fn emit_bloq_stim_rejects_discard_node() {
        // A `Discard` is shot postselection the static backend cannot express, so
        // it is rejected up front rather than silently dropped (IR spec §6).
        let mut bloq = Bloq::new();
        bloq.add_node(BloqNode::classical(ClassicalNode::Discard {
            condition: ClassicalExpr::Const(true),
        }));
        assert_eq!(
            emit_bloq_stim(&bloq),
            Err(StimEmissionError::UnsupportedNode(
                "Discard (shot postselection)"
            ))
        );
    }

    #[test]
    fn emit_bloq_stim_skips_records_reducible_classical_node() {
        // A records-reducible classical node passes the feature gate (no producer
        // emits it yet) and must be skipped on the static path, not panic the
        // global pre-passes in prepare() by calling quantum() on it.
        let mut bloq = Bloq::new();
        let input = bloq.add_node(BloqNode::classical(ClassicalNode::observable_fragment(
            vec![],
            vec![],
        )));
        let compute = bloq.add_node(BloqNode::classical(ClassicalNode::Compute {
            expr: ClassicalExpr::In(0),
        }));
        bloq.add_edge(input, compute, BloqEdge::value(0));
        emit_bloq_stim(&bloq).unwrap();
    }

    #[test]
    fn emit_bloq_stim_audits_only_when_requested() {
        let mut bloq = Bloq::new();
        bloq.add_node(BloqNode::classical(ClassicalNode::Compute {
            expr: ClassicalExpr::In(0),
        }));

        // The invalid classical input is irrelevant to the circuit emitted
        // here, but the full T2 audit must still catch it when requested.
        emit_bloq_stim(&bloq).unwrap();
        emit_bloq_stim_segments(&bloq).unwrap();
        emit_bloq_stim_with(&bloq, &BloqStimOptions::new().with_align_moments(true)).unwrap();

        assert!(matches!(
            emit_bloq_stim_with(
                &bloq,
                &BloqStimOptions::new().with_trust(InputTrust::Checked)
            ),
            Err(StimEmissionError::InvalidProgram(
                bloq_ir::BloqValidationError::MissingClassicalInput { slot: 0, .. }
            ))
        ));
        assert!(matches!(
            emit_bloq_stim_with(
                &bloq,
                &BloqStimOptions::new()
                    .with_trust(InputTrust::Checked)
                    .with_align_moments(true)
            ),
            Err(StimEmissionError::InvalidProgram(
                bloq_ir::BloqValidationError::MissingClassicalInput { slot: 0, .. }
            ))
        ));
    }

    #[test]
    fn static_emission_rejects_unpinned_membership_before_building_plans() {
        let mut bloq = Bloq::new();
        let mut node = BloqNode::from_members(Vec::new());
        node.expect_quantum_mut()
            .guards
            .push(bloq_ir::QuantumGuard {
                input: 0,
                ..Default::default()
            });
        bloq.add_node(node);

        assert_eq!(
            emit_bloq_stim(&bloq),
            Err(StimEmissionError::UnsupportedNode(
                "conditional component membership"
            ))
        );
        assert_eq!(
            emit_bloq_stim_segments(&bloq).unwrap_err(),
            StimEmissionError::UnsupportedNode("conditional component membership")
        );
    }

    #[test]
    fn aligned_emission_rejects_noise_and_segments() {
        let bloq = Bloq::new();
        let noise = NoiseModel::uniform_depolarizing(1e-3);
        let options = BloqStimOptions::new().with_align_moments(true);

        assert_eq!(
            emit_bloq_stim_with(&bloq, &options.with_noise(&noise)),
            Err(StimEmissionError::AlignMomentsWithNoise)
        );
        assert!(matches!(
            emit_bloq_stim_segments_with(&bloq, &options),
            Err(StimEmissionError::AlignMomentsWithSegments)
        ));
    }

    #[test]
    fn attempt_emission_preserves_typed_flatten_resource_errors() {
        let mut circuit = CoordCircuit::new();
        let body = circuit.add_body(CircuitBody::from_ops(vec![bloq_circuit::Op::Tick]));
        circuit.push_repeat(body, u32::MAX);
        let mut program = Bloq::new();
        let template = program.add_template(BloqTemplate::new(circuit));
        let mut node = BloqNode::from_members(Vec::new());
        node.expect_quantum_mut()
            .instances
            .push(TemplateInstance::new(
                TemplateInstanceId(0),
                template,
                ivec2(0, 0),
            ));
        program.add_node(node);

        let error = emit_isolated_t_attempts(&program, 0.0).unwrap_err();
        assert!(std::error::Error::source(&error).is_some());
        assert!(matches!(error, StimEmissionError::Flatten(error) if error.is_resource_limited()));
    }

    #[test]
    fn emit_bloq_stim_validates_before_capability_checks() {
        let mut bloq = Bloq::new();
        bloq.add_node(BloqNode::region(RegionNode::RepeatUntilSuccess {
            restart_condition: ClassicalExpr::In(0),
            restart_source: None,
            body: SubGraph::new(),
        }));

        assert!(matches!(
            emit_bloq_stim_with(
                &bloq,
                &BloqStimOptions::new().with_trust(InputTrust::Checked)
            ),
            Err(StimEmissionError::InvalidProgram(
                bloq_ir::BloqValidationError::MissingRegionSelectorInput { slot: 0, .. }
            ))
        ));
        assert!(matches!(
            emit_bloq_stim_segments_with(
                &bloq,
                &BloqStimOptions::new().with_trust(InputTrust::Checked)
            ),
            Err(StimEmissionError::InvalidProgram(
                bloq_ir::BloqValidationError::MissingRegionSelectorInput { slot: 0, .. }
            ))
        ));
    }

    #[test]
    fn emit_bloq_stim_rejects_nonlinear_compute() {
        // A nonlinear (`And`/`Or`) feedback condition is not XOR-expressible in
        // `OBSERVABLE_INCLUDE`, so the static backend rejects it up front (it needs
        // conditional execution, Phase U6/U7). Linear conditions fold instead.
        let mut bloq = Bloq::new();
        let compute = bloq.add_node(BloqNode::classical(ClassicalNode::Compute {
            expr: ClassicalExpr::And(Box::new([ClassicalExpr::In(0), ClassicalExpr::In(1)])),
        }));
        // Wire dummy inputs so the node is a genuine two-input `And`.
        let a = bloq.add_node(BloqNode::classical(ClassicalNode::observable_fragment(
            vec![],
            vec![],
        )));
        let b = bloq.add_node(BloqNode::classical(ClassicalNode::observable_fragment(
            vec![],
            vec![],
        )));
        bloq.add_edge(a, compute, BloqEdge::value(0));
        bloq.add_edge(b, compute, BloqEdge::value(1));
        assert_eq!(
            emit_bloq_stim(&bloq),
            Err(StimEmissionError::UnsupportedNode(
                "Compute (nonlinear And/Or condition requires conditional execution)"
            ))
        );
    }

    #[test]
    fn shared_classical_dag_emits_canonical_record_parity() {
        let mut graph = Bloq::new();
        let node = measured_program_node(
            &mut graph,
            ivec3(0, 0, 0),
            ivec2(0, 0),
            TemplateInstanceId(0),
        );
        let quantum = graph.add_node(node);
        let measured = graph.add_node(BloqNode::classical(ClassicalNode::observable_fragment(
            vec![InstanceMeasurement {
                instance: TemplateInstanceId(0),
                measurement: 0,
            }],
            vec![],
        )));
        graph.add_edge(quantum, measured, BloqEdge::Order);
        let mut previous = measured;
        for _ in 0..25 {
            let next = graph.add_node(BloqNode::classical(ClassicalNode::Compute {
                expr: ClassicalExpr::Xor(Box::new([ClassicalExpr::In(0), ClassicalExpr::In(1)])),
            }));
            graph.add_edge(previous, next, BloqEdge::value(0));
            graph.add_edge(previous, next, BloqEdge::value(1));
            previous = next;
        }
        let zero = graph.add_node(BloqNode::classical(ClassicalNode::observable(0)));
        graph.add_edge(previous, zero, BloqEdge::value(0));
        let nonzero = graph.add_node(BloqNode::classical(ClassicalNode::observable(1)));
        graph.add_edge(previous, nonzero, BloqEdge::value(0));
        graph.add_edge(measured, nonzero, BloqEdge::value(1));
        let text = emit_bloq_stim(&graph).unwrap();
        let observables = text
            .lines()
            .filter(|line| line.starts_with("OBSERVABLE_INCLUDE"))
            .collect::<Vec<_>>();
        assert_eq!(
            observables,
            ["OBSERVABLE_INCLUDE(0)", "OBSERVABLE_INCLUDE(1) rec[-1]"]
        );
    }

    #[test]
    fn emit_bloq_stim_folds_linear_compute_into_observable() {
        // A linear feedback condition folds into the affected observable: the
        // `Not(In0)` `Compute` over a measurement fragment, routed into
        // observable 1, contributes that measurement's record to
        // `OBSERVABLE_INCLUDE(1)` (the `Not` sign is dropped — Stim is
        // sign-agnostic). This exercises the recursive fold through a `Compute`.
        let mut graph = Bloq::new();
        let parent_node = measured_program_node(
            &mut graph,
            ivec3(0, 0, 0),
            ivec2(0, 0),
            TemplateInstanceId(1),
        );
        let qubit = ivec2(0, 0);
        let mut template_circuit = CoordCircuit::new();
        template_circuit.measure(PauliBasis::Z, [qubit]);
        let template = graph.add_template(BloqTemplate::new(template_circuit));
        let mut node = BloqNode::from_members(vec![SourceBlockRef {
            pos: ivec3(0, 0, 0),
        }]);
        node.expect_quantum_mut()
            .instances
            .push(TemplateInstance::new(
                TemplateInstanceId(0),
                template,
                ivec2(1, 0),
            ));
        let parent = graph.add_node(parent_node);
        let child = graph.add_node(node);
        graph.add_edge(parent, child, BloqEdge::quantum(vec![]));

        let accumulate = graph.add_node(BloqNode::classical(ClassicalNode::observable_fragment(
            vec![InstanceMeasurement {
                instance: TemplateInstanceId(0),
                measurement: 0,
            }],
            vec![],
        )));
        graph.add_edge(child, accumulate, BloqEdge::Order);

        // The feedback condition `!m` lowers to a `Not` `Compute` over the
        // measurement, routed into a downstream observable it perturbs.
        let compute = graph.add_node(BloqNode::classical(ClassicalNode::Compute {
            expr: ClassicalExpr::Not(Box::new(ClassicalExpr::In(0))),
        }));
        graph.add_edge(accumulate, compute, BloqEdge::value(0));
        let observable = graph.add_node(BloqNode::classical(ClassicalNode::observable(1)));
        graph.add_edge(compute, observable, BloqEdge::value(0));

        let text = emit_bloq_stim(&graph).expect("emit Bloq Stim");
        assert!(
            text.contains("OBSERVABLE_INCLUDE(1) rec[-1]"),
            "linear Compute condition should fold its measurement record into the observable: {text}"
        );
    }

    /// A template-backed node whose single instance measures `qubit` in Z.
    ///
    /// Each call registers its own template and must be given a graph-unique
    /// `instance_id`; the template measures at the origin and the instance is
    /// offset to `qubit`.
    fn measured_program_node(
        graph: &mut Bloq,
        block_pos: glam::IVec3,
        qubit: glam::IVec2,
        instance_id: TemplateInstanceId,
    ) -> BloqNode {
        let mut circuit = CoordCircuit::new();
        circuit.measure(PauliBasis::Z, [ivec2(0, 0)]);
        template_node_with_circuit_at(graph, circuit, block_pos, instance_id, qubit)
    }

    fn template_node_with_circuit(graph: &mut Bloq, circuit: CoordCircuit) -> BloqNode {
        template_node_with_circuit_at(
            graph,
            circuit,
            ivec3(0, 0, 0),
            TemplateInstanceId(0),
            IVec2::ZERO,
        )
    }

    fn template_node_with_circuit_at(
        graph: &mut Bloq,
        circuit: CoordCircuit,
        block_pos: glam::IVec3,
        instance_id: TemplateInstanceId,
        offset: IVec2,
    ) -> BloqNode {
        let template = graph.add_template(BloqTemplate::new(circuit));
        let mut node = BloqNode::from_members(vec![SourceBlockRef { pos: block_pos }]);
        node.expect_quantum_mut()
            .instances
            .push(TemplateInstance::new(instance_id, template, offset));
        node
    }

    fn aligned_round_circuit(interaction_slots: usize) -> CoordCircuit {
        let data = ivec2(0, 0);
        let ancilla = ivec2(1, 0);
        let mut circuit = CoordCircuit::new();
        circuit.do_gate(GateType::RZ, [data, ancilla]).unwrap();
        circuit.tick();
        for _ in 0..interaction_slots {
            circuit.do_gate(GateType::CX, [data, ancilla]).unwrap();
            circuit.tick();
        }
        circuit.measure(PauliBasis::Z, [data, ancilla]);
        circuit
    }

    fn aligned_rotation_rounds(rounds: usize) -> CoordCircuit {
        let qubit = ivec2(0, 0);
        let mut circuit = CoordCircuit::new();
        for round in 0..rounds {
            circuit.do_gate(GateType::RZ, [qubit]).unwrap();
            circuit.tick();
            circuit.do_gate(GateType::H, [qubit]).unwrap();
            circuit.tick();
            circuit.measure(PauliBasis::Z, [qubit]);
            if round + 1 != rounds {
                circuit.tick();
            }
        }
        circuit
    }

    #[test]
    fn aligned_emission_merges_same_layer_slots_and_keeps_record_lookbacks() {
        let mut graph = Bloq::new();
        let mut first_quantum = None;
        for (instance, interaction_slots, offset) in [
            (TemplateInstanceId(0), 4, ivec2(0, 0)),
            (TemplateInstanceId(1), 5, ivec2(10, 0)),
            (TemplateInstanceId(2), 6, ivec2(20, 0)),
        ] {
            let mut template = BloqTemplate::new(aligned_round_circuit(interaction_slots));
            template.detectors.push(TemplateDetector {
                scope: TemplateDetectorScope::TopLevel,
                parity: DetectorParity::from_measurements([0]),
                coords: None,
            });
            let template = graph.add_template(template);
            let mut node = BloqNode::from_members(vec![SourceBlockRef {
                pos: ivec3(offset.x, 0, 0),
            }]);
            node.expect_quantum_mut()
                .instances
                .push(TemplateInstance::new(instance, template, offset));
            first_quantum.get_or_insert(graph.add_node(node));
        }
        let accumulate = graph.add_node(BloqNode::classical(ClassicalNode::observable_fragment(
            vec![InstanceMeasurement {
                instance: TemplateInstanceId(0),
                measurement: 0,
            }],
            vec![],
        )));
        let observable = graph.add_node(BloqNode::classical(ClassicalNode::observable(7)));
        graph.add_edge(first_quantum.unwrap(), accumulate, BloqEdge::Order);
        graph.add_edge(accumulate, observable, BloqEdge::compose(0));

        let text = emit_bloq_stim_with(&graph, &BloqStimOptions::new().with_align_moments(true))
            .expect("emit aligned Bloq Stim");
        let slots = text.split("TICK\n").collect::<Vec<_>>();

        assert_eq!(
            slots.len() - 1,
            8,
            "four and five CX slots pad into six: {text}"
        );
        assert!(slots[0].contains("R 0 1"), "{text}");
        assert!(slots[0].contains("R 2 3"), "{text}");
        assert!(slots[0].contains("R 4 5"), "{text}");
        assert!(slots[7].contains("M 0 1"), "{text}");
        assert!(slots[7].contains("M 2 3"), "{text}");
        assert!(slots[7].contains("M 4 5"), "{text}");
        assert_eq!(text.matches("DETECTOR rec[-2]").count(), 3, "{text}");
        assert!(
            text.contains("OBSERVABLE_INCLUDE(7) rec[-6]"),
            "global frame must include every contributor: {text}"
        );
    }

    #[test]
    fn aligned_emission_projects_timeline_rounds_into_source_layers() {
        let mut graph = Bloq::new();
        let mut tall = template_node_with_circuit_at(
            &mut graph,
            aligned_rotation_rounds(2),
            ivec3(0, 0, 2),
            TemplateInstanceId(0),
            ivec2(0, 0),
        );
        tall.expect_quantum_mut().timeline = Some(QuantumTimeline {
            layer_round_ends: vec![1, 2],
        });
        let tall = graph.add_node(tall);
        let upper = template_node_with_circuit_at(
            &mut graph,
            aligned_rotation_rounds(1),
            ivec3(1, 0, 3),
            TemplateInstanceId(1),
            ivec2(10, 0),
        );
        let upper = graph.add_node(upper);

        let text = emit_bloq_stim_with(&graph, &BloqStimOptions::new().with_align_moments(true))
            .expect("emit timeline-aligned Bloq Stim");
        let slots = text.split("TICK\n").collect::<Vec<_>>();

        assert_eq!(
            slots.len() - 1,
            6,
            "second tall round shares upper layer: {text}"
        );
        assert!(slots[0].contains(&format!("# node {}:", tall.0)), "{text}");
        assert!(slots[3].contains("R 0"), "{text}");
        assert!(slots[3].contains(&format!("# node {}:", upper.0)), "{text}");
        assert!(slots[3].contains("R 1"), "{text}");
    }

    #[test]
    fn aligned_terminal_tick_keeps_detector_boundary_without_phantom_moment() {
        let mut circuit = CoordCircuit::new();
        circuit.measure(PauliBasis::Z, [ivec2(0, 0)]);
        circuit.tick();
        let mut graph = Bloq::new();
        let mut template = BloqTemplate::new(circuit);
        template.detectors.push(TemplateDetector {
            scope: TemplateDetectorScope::TopLevel,
            parity: DetectorParity::from_measurements([0]),
            coords: None,
        });
        let template = graph.add_template(template);
        let mut node = BloqNode::from_members(vec![SourceBlockRef {
            pos: ivec3(0, 0, 0),
        }]);
        node.expect_quantum_mut()
            .instances
            .push(TemplateInstance::new(
                TemplateInstanceId(0),
                template,
                IVec2::ZERO,
            ));
        graph.add_node(node);

        let ordinary = emit_bloq_stim(&graph).unwrap();
        let aligned =
            emit_bloq_stim_with(&graph, &BloqStimOptions::new().with_align_moments(true)).unwrap();

        assert_eq!(aligned, ordinary);
        assert_eq!(aligned.matches("TICK\n").count(), 2, "{aligned}");
        assert!(aligned.contains("M 0\nTICK\nDETECTOR rec[-1]\nTICK\n"));
    }

    #[test]
    fn aligned_empty_circuit_is_silent_or_rejects_unplaceable_metadata() {
        let mut graph = Bloq::new();
        let node = template_node_with_circuit(&mut graph, CoordCircuit::new());
        let node = graph.add_node(node);
        let options = BloqStimOptions::new().with_align_moments(true);

        assert_eq!(
            emit_bloq_stim_with(&graph, &options).unwrap(),
            emit_bloq_stim(&graph).unwrap()
        );

        graph
            .node_mut(node)
            .unwrap()
            .expect_quantum_mut()
            .detectors
            .push(NodeDetector {
                parity: DetectorParity::<InstanceMeasurement>::default().with_sign(true),
                coords: None,
            });
        let error = emit_bloq_stim_with(&graph, &options).unwrap_err();
        assert_eq!(error, StimEmissionError::AlignMomentsWithEmptyMetadata);
    }

    #[test]
    fn emit_bloq_stim_reports_unsupported_gate() {
        let qubit = ivec2(0, 0);
        let mut circuit = CoordCircuit::new();
        circuit.do_gate(GateType::T, [qubit]).unwrap();
        let mut graph = Bloq::new();
        let node = template_node_with_circuit(&mut graph, circuit);
        graph.add_node(node);

        let error = emit_bloq_stim(&graph).unwrap_err();

        assert_eq!(error, StimEmissionError::UnsupportedGate(GateType::T));
    }

    #[test]
    fn unused_non_clifford_template_does_not_block_emission() {
        let qubit = ivec2(0, 0);
        let mut unused = CoordCircuit::new();
        unused.do_gate(GateType::T, [qubit]).unwrap();
        let mut graph = Bloq::new();
        graph.add_template(BloqTemplate::new(unused));

        let mut used = CoordCircuit::new();
        used.do_gate(GateType::H, [qubit]).unwrap();
        let node = template_node_with_circuit(&mut graph, used);
        graph.add_node(node);

        let text = emit_bloq_stim(&graph).expect("unused templates are not emitted");
        assert!(text.lines().any(|line| line == "H 0"), "{text}");
    }

    #[test]
    fn unreachable_non_clifford_body_does_not_block_emission() {
        let qubit = ivec2(0, 0);
        let mut circuit = CoordCircuit::new();
        circuit.do_gate(GateType::H, [qubit]).unwrap();
        circuit.add_body(CircuitBody::from_ops(vec![Op::Gate {
            gate: GateType::T,
            qubits: vec![qubit],
        }]));
        let mut graph = Bloq::new();
        let node = template_node_with_circuit(&mut graph, circuit);
        graph.add_node(node);

        let text = emit_bloq_stim(&graph).expect("unreachable bodies are not emitted");
        assert!(text.lines().any(|line| line == "H 0"), "{text}");
    }

    #[test]
    fn include_boundary_face_fans_out_to_every_observable() {
        let qubit = ivec2(0, 0);
        let instance = TemplateInstanceId(0);
        for nested in [false, true] {
            let mut circuit = CoordCircuit::new();
            circuit.do_gate(GateType::X, [qubit]).unwrap();
            let mut graph = Bloq::new();
            let quantum = template_node_with_circuit(&mut graph, circuit);
            graph.add_node(quantum);
            let include = graph.add_node(BloqNode::classical(ClassicalNode::observable_fragment(
                vec![],
                vec![InstanceBoundaryOperator {
                    instance,
                    face: BoundaryFace::Output,
                    operator: [(qubit, bloq_circuit::Pauli::Z)].into_iter().collect(),
                }],
            )));
            let producer = if nested {
                let recipe = graph.add_node(BloqNode::classical(
                    ClassicalNode::observable_fragment(vec![], vec![]),
                ));
                graph.add_edge(include, recipe, BloqEdge::compose(0));
                recipe
            } else {
                include
            };
            for index in [0, 1] {
                let observable =
                    graph.add_node(BloqNode::classical(ClassicalNode::observable(index)));
                graph.add_edge(producer, observable, BloqEdge::compose(0));
            }

            let text = emit_bloq_stim(&graph).expect("shared fragment emits for both consumers");
            let aligned =
                emit_bloq_stim_with(&graph, &BloqStimOptions::new().with_align_moments(true))
                    .unwrap();
            for output in [&text, &aligned] {
                for index in [0, 1] {
                    assert!(
                        output.contains(&format!("OBSERVABLE_INCLUDE({index}) Z0")),
                        "{output}"
                    );
                }
            }
            assert!(aligned.contains("X 0\nTICK\nOBSERVABLE_INCLUDE(0) Z0"));
            if nested {
                for enabled in [false, true] {
                    let mut guarded = graph.clone();
                    let gate = guarded.add_node(BloqNode::classical(ClassicalNode::Compute {
                        expr: ClassicalExpr::Const(enabled),
                    }));
                    guarded.node_mut(producer).unwrap().activation = Some(1);
                    guarded.add_edge(gate, producer, BloqEdge::value(1));
                    assert_eq!(
                        emit_bloq_stim(&guarded),
                        Err(StimEmissionError::UnsupportedNode(
                            "activation (conditional execution)"
                        ))
                    );
                }
            }
        }
    }

    #[test]
    fn indexed_composition_includes_boundaries_but_value_ports_do_not() {
        let qubit = ivec2(0, 0);
        let mut circuit = CoordCircuit::new();
        circuit.do_gate(GateType::X, [qubit]).unwrap();
        let mut graph = Bloq::new();
        let quantum = template_node_with_circuit(&mut graph, circuit);
        let quantum = graph.add_node(quantum);
        let child = graph.add_node(BloqNode::classical(ClassicalNode::Observable {
            index: Some(0),
            measurements: vec![],
            operators: vec![InstanceBoundaryOperator {
                instance: TemplateInstanceId(0),
                face: BoundaryFace::Output,
                operator: [(qubit, Pauli::Z)].into_iter().collect(),
            }],
        }));
        graph.add_edge(quantum, child, BloqEdge::Order);
        for (index, edge) in [
            (1, BloqEdge::compose(0)),
            (2, BloqEdge::value(0)),
            (3, BloqEdge::flip(0)),
        ] {
            let parent = graph.add_node(BloqNode::classical(ClassicalNode::observable(index)));
            graph.add_edge(child, parent, edge);
        }
        let text = emit_bloq_stim(&graph).unwrap();
        for index in [0, 1] {
            assert!(
                text.contains(&format!("OBSERVABLE_INCLUDE({index}) Z0")),
                "{text}"
            );
        }
        for index in [2, 3] {
            assert!(
                !text.contains(&format!("OBSERVABLE_INCLUDE({index}) Z0")),
                "{text}"
            );
        }
    }

    #[test]
    fn repeated_recipe_paths_cancel_boundary_faces() {
        let qubit = ivec2(0, 0);
        let mut circuit = CoordCircuit::new();
        circuit.do_gate(GateType::X, [qubit]).unwrap();
        let mut graph = Bloq::new();
        let quantum = template_node_with_circuit(&mut graph, circuit);
        graph.add_node(quantum);
        let mut root = graph.add_node(BloqNode::classical(ClassicalNode::observable_fragment(
            vec![],
            vec![InstanceBoundaryOperator {
                instance: TemplateInstanceId(0),
                face: BoundaryFace::Output,
                operator: [(qubit, bloq_circuit::Pauli::Z)].into_iter().collect(),
            }],
        )));
        for _ in 0..24 {
            let next = graph.add_node(BloqNode::classical(ClassicalNode::observable_fragment(
                vec![],
                vec![],
            )));
            graph.add_edge(root, next, BloqEdge::compose(0));
            graph.add_edge(root, next, BloqEdge::compose(1));
            root = next;
        }
        let observable = graph.add_node(BloqNode::classical(ClassicalNode::observable(0)));
        graph.add_edge(root, observable, BloqEdge::compose(0));
        let text = emit_bloq_stim(&graph).unwrap();
        assert!(!text.contains("OBSERVABLE_INCLUDE(0) Z0"), "{text}");
    }

    #[test]
    fn emit_bloq_stim_instantiates_template_node_side_tables() {
        let mut graph = Bloq::new();
        let parent_node = measured_program_node(
            &mut graph,
            ivec3(0, 0, 0),
            ivec2(0, 0),
            TemplateInstanceId(1),
        );
        let qubit = ivec2(0, 0);
        let mut template_circuit = CoordCircuit::new();
        template_circuit.measure(PauliBasis::Z, [qubit]);
        let mut bloq_template = BloqTemplate::new(template_circuit);
        bloq_template.detectors.push(TemplateDetector {
            scope: TemplateDetectorScope::TopLevel,
            parity: DetectorParity::from_measurements([0]),
            coords: Some(bloq_circuit::DetectorCoords::from_slice(&[3.0, 4.0])),
        });
        let template = graph.add_template(bloq_template);
        let mut node = BloqNode::from_members(vec![SourceBlockRef {
            pos: ivec3(0, 0, 0),
        }]);
        node.expect_quantum_mut()
            .instances
            .push(TemplateInstance::new(
                TemplateInstanceId(0),
                template,
                ivec2(1, 0),
            ));
        let parent = graph.add_node(parent_node);
        let child = graph.add_node(node);
        graph.add_edge(parent, child, BloqEdge::quantum(vec![]));

        // The observable composes a measurement fragment, with a
        // read-after-measure Order edge from the producing node (IR spec §4/§5).
        let accumulate = graph.add_node(BloqNode::classical(ClassicalNode::observable_fragment(
            vec![InstanceMeasurement {
                instance: TemplateInstanceId(0),
                measurement: 0,
            }],
            vec![],
        )));
        let observable = graph.add_node(BloqNode::classical(ClassicalNode::observable(0)));
        graph.add_edge(child, accumulate, BloqEdge::Order);
        graph.add_edge(accumulate, observable, BloqEdge::compose(0));

        let text = emit_bloq_stim(&graph).expect("emit Bloq Stim");

        insta::assert_snapshot!(text, @"
        QUBIT_COORDS(0, 0) 0
        QUBIT_COORDS(1, 0) 1
        # node 0: quantum blocks (0,0,0)
        M 0
        TICK
        # node 1: quantum blocks (0,0,0)
        M 1
        DETECTOR(4, 4) rec[-1]
        TICK
        # node 3: observable 0 synthetic
        OBSERVABLE_INCLUDE(0) rec[-1]
        ");
    }

    #[test]
    fn idle_noise_expands_repeat_annotations_and_preserves_segment_records() {
        let (a, b) = (ivec2(0, 0), ivec2(1, 0));
        let mut circuit = CoordCircuit::new();
        circuit.do_gate(GateType::RZ, [a]).unwrap();
        let measurement = circuit.reserve_measurement_id(b);
        let body = circuit.add_body(CircuitBody::from_ops(vec![
            Op::Measure {
                basis: PauliBasis::Z,
                qubits: vec![b],
                measurements: vec![measurement],
                flip_probability: 0.0,
            },
            Op::Tick,
        ]));
        circuit.push_repeat(body, 3);
        let mut template = BloqTemplate::new(circuit);
        template.detectors.push(TemplateDetector {
            scope: TemplateDetectorScope::RepeatBody { body },
            parity: DetectorParity::from_terms([
                DetectorTerm::Measurement(measurement),
                DetectorTerm::LoopState(LoopStateId(0)),
            ]),
            coords: None,
        });
        template.repeat_states.push(TemplateRepeatState {
            body,
            state: LoopStateId(0),
            initial: DetectorParity::default(),
            next: DetectorParity::from_measurements([measurement]),
        });
        template.restarts.push(bloq_ir::lowering::TemplateRestart {
            parity: DetectorParity::from_terms([DetectorTerm::LoopState(LoopStateId(0))]),
        });
        let mut without_restart = template.clone();
        without_restart.restarts.clear();
        let mut program = Bloq::new();
        let template_id = program.add_template(template);
        let mut node = BloqNode::from_members(vec![]);
        node.expect_quantum_mut()
            .instances
            .push(TemplateInstance::new(
                TemplateInstanceId(0),
                template_id,
                IVec2::ZERO,
            ));
        let noise = NoiseModel {
            p_idle: 0.125,
            ..NoiseModel::uniform_depolarizing(0.0)
        };
        let plan = node
            .emission_plan_with_options(program.templates(), InstantiationOptions::noisy(&noise))
            .unwrap();
        let normalized = &plan.templates(program.templates())[template_id];
        assert_eq!(normalized.detectors.len(), 3);
        assert!(normalized.repeat_states.is_empty());
        assert_eq!(
            normalized.restarts[0].parity,
            DetectorParity::from_measurements([measurement])
        );
        assert_eq!(
            plan.circuit.expanded_measurement_columns().unwrap().count(),
            3
        );

        // Static Stim emission excludes RUS-only restart annotations; their
        // expansion was checked above on the same normalized source template.
        node.expect_quantum_mut().instances[0].template_id = program.add_template(without_restart);
        let producer = program.add_node(node);
        let record = program.add_node(BloqNode::classical(ClassicalNode::observable_fragment(
            vec![InstanceMeasurement {
                instance: TemplateInstanceId(0),
                measurement,
            }],
            vec![],
        )));
        let observable = program.add_node(BloqNode::classical(ClassicalNode::observable(0)));
        program.add_edge(producer, record, BloqEdge::Order);
        program.add_edge(record, observable, BloqEdge::compose(0));
        let (clean, noisy) = emit_bloq_stim_segments_pair(&program, &noise).unwrap();
        assert_eq!(clean.num_measurements(), 3);
        assert_eq!(noisy.num_measurements(), 3);
        assert_eq!(
            noisy.to_text().matches("DEPOLARIZE1(0.125) 0").count(),
            2,
            "the leading reset occupies a's first moment, but a idles on later iterations"
        );
        assert_eq!(noisy.to_text().matches("DETECTOR").count(), 3);
        for text in [clean.to_text(), noisy.to_text()] {
            assert!(
                text.contains("OBSERVABLE_INCLUDE(0) rec[-1]"),
                "the original measurement id still names the final loop record: {text}"
            );
        }
        let mut flat = program.clone();
        flat.flatten().unwrap();
        let expected =
            emit_bloq_stim_with(&flat, &BloqStimOptions::new().with_noise(&noise)).unwrap();
        assert_eq!(noisy.to_text(), expected);
        #[cfg(feature = "verify")]
        for text in [clean.to_text(), noisy.to_text()] {
            let circuit: stim::Circuit = text.parse().unwrap();
            assert_eq!(circuit.num_measurements(), 3);
            assert_eq!(circuit.num_detectors(), 3);
        }
    }

    #[test]
    fn emit_bloq_stim_instantiates_template_repeat_side_tables() {
        let qubit = ivec2(0, 0);
        let mut template_circuit = CoordCircuit::new();
        let body = template_circuit.add_body(CircuitBody::from_ops(vec![Op::Measure {
            basis: PauliBasis::Z,
            qubits: vec![qubit],
            measurements: vec![0],
            flip_probability: 0.0,
        }]));
        template_circuit.register_measurement_id(0, qubit);
        template_circuit.push_repeat(body, 2);
        let mut bloq_template = BloqTemplate::new(template_circuit);
        bloq_template.detectors.push(TemplateDetector {
            scope: TemplateDetectorScope::RepeatBody { body },
            parity: DetectorParity::from_measurements([0]),
            coords: None,
        });
        bloq_template.repeat_states.push(TemplateRepeatState {
            body,
            state: LoopStateId(0),
            initial: DetectorParity::default(),
            next: DetectorParity::from_measurements([0]),
        });
        let mut graph = Bloq::new();
        let template = graph.add_template(bloq_template);
        let mut node = BloqNode::from_members(vec![SourceBlockRef {
            pos: ivec3(0, 0, 0),
        }]);
        node.expect_quantum_mut()
            .instances
            .push(TemplateInstance::new(
                TemplateInstanceId(0),
                template,
                qubit,
            ));
        graph.add_node(node);

        let text = emit_bloq_stim(&graph).expect("emit Bloq Stim");

        insta::assert_snapshot!(text, @"
        QUBIT_COORDS(0, 0) 0
        # node 0: quantum blocks (0,0,0)
        REPEAT 2 {
            M 0
            DETECTOR rec[-1]
        }
        ");
    }

    #[test]
    fn emit_bloq_stim_emits_node_circuits_in_topological_order() {
        let mut graph = Bloq::new();
        let parent_node = measured_program_node(
            &mut graph,
            ivec3(0, 0, 0),
            ivec2(0, 0),
            TemplateInstanceId(0),
        );
        let child_node = measured_program_node(
            &mut graph,
            ivec3(0, 0, 1),
            ivec2(1, 0),
            TemplateInstanceId(1),
        );
        // Insert opposite to dependency order so storage order cannot pass.
        let child = graph.add_node(child_node);
        let parent = graph.add_node(parent_node);
        graph.add_edge(parent, child, BloqEdge::quantum(vec![]));

        let text = emit_bloq_stim(&graph).expect("emit Bloq Stim");

        let measurement_lines = text
            .lines()
            .filter(|line| line.starts_with("M "))
            .collect::<Vec<_>>();
        assert_eq!(measurement_lines, vec!["M 0", "M 1"], "{text}");
    }

    #[test]
    fn emit_bloq_stim_resolves_cross_node_detector_refs() {
        let mut graph = Bloq::new();
        let mut parent_circuit = CoordCircuit::new();
        parent_circuit.measure(PauliBasis::Z, [ivec2(0, 0)]);
        let parent_node = template_node_with_circuit_at(
            &mut graph,
            parent_circuit,
            ivec3(0, 0, 0),
            TemplateInstanceId(0),
            IVec2::ZERO,
        );
        let mut child_circuit = CoordCircuit::new();
        child_circuit.measure(PauliBasis::Z, [ivec2(1, 0)]);
        let mut child_node = template_node_with_circuit_at(
            &mut graph,
            child_circuit,
            ivec3(0, 0, 1),
            TemplateInstanceId(1),
            IVec2::ZERO,
        );
        child_node
            .expect_quantum_mut()
            .detectors
            .push(NodeDetector {
                parity: DetectorParity::from_measurements([
                    InstanceMeasurement {
                        instance: TemplateInstanceId(0),
                        measurement: 0,
                    },
                    InstanceMeasurement {
                        instance: TemplateInstanceId(1),
                        measurement: 0,
                    },
                ]),
                coords: Some(bloq_circuit::DetectorCoords::from_slice(&[5.0, 6.0])),
            });

        let parent = graph.add_node(parent_node);
        let child = graph.add_node(child_node);
        graph.add_edge(parent, child, BloqEdge::quantum(vec![]));

        let text = emit_bloq_stim(&graph).expect("emit Bloq Stim");

        assert!(text.contains("DETECTOR(5, 6) rec[-2] rec[-1]"), "{text}");
    }

    #[test]
    fn emit_bloq_stim_resolves_shared_cross_node_detector() {
        use bloq_ir::{BundleDetector, BundleMeasurement, DetectorBundle, DetectorBundleUse};

        let mut graph = Bloq::new();
        let mut parent_circuit = CoordCircuit::new();
        parent_circuit.measure(PauliBasis::Z, [ivec2(0, 0)]);
        let parent_node = template_node_with_circuit_at(
            &mut graph,
            parent_circuit,
            ivec3(0, 0, 0),
            TemplateInstanceId(0),
            IVec2::ZERO,
        );
        let mut child_circuit = CoordCircuit::new();
        child_circuit.measure(PauliBasis::Z, [ivec2(1, 0)]);
        let mut child_node = template_node_with_circuit_at(
            &mut graph,
            child_circuit,
            ivec3(0, 0, 1),
            TemplateInstanceId(1),
            IVec2::ZERO,
        );
        let owner_templates = vec![
            parent_node.expect_quantum().instances[0].template_id,
            child_node.expect_quantum().instances[0].template_id,
        ];
        let bundle = graph.add_detector_bundle(DetectorBundle::new(
            owner_templates,
            vec![BundleDetector {
                parity: bloq_circuit::DetectorParity::from_measurements([
                    BundleMeasurement {
                        owner: 0,
                        measurement: 0,
                    },
                    BundleMeasurement {
                        owner: 1,
                        measurement: 0,
                    },
                ]),
                coords: Some(bloq_circuit::DetectorCoords::from_slice(&[1.0, 2.0])),
            }],
        ));
        child_node
            .expect_quantum_mut()
            .detector_bundles
            .push(DetectorBundleUse {
                bundle,
                instances: vec![TemplateInstanceId(0), TemplateInstanceId(1)],
                offset: IVec2::new(4, 4),
            });

        let parent = graph.add_node(parent_node);
        let child = graph.add_node(child_node);
        graph.add_edge(parent, child, BloqEdge::quantum(vec![]));

        let text = emit_bloq_stim(&graph).expect("emit shared detector");
        assert!(text.contains("DETECTOR(5, 6) rec[-2] rec[-1]"), "{text}");

        let wrong_bundle = graph.add_detector_bundle(DetectorBundle::new(
            vec![graph[child].expect_quantum().instances[0].template_id; 2],
            vec![],
        ));
        graph
            .node_mut(child)
            .unwrap()
            .expect_quantum_mut()
            .detector_bundles[0]
            .bundle = wrong_bundle;
        assert!(matches!(
            emit_bloq_stim(&graph),
            Err(StimEmissionError::DetectorBundle(
                bloq_ir::DetectorBundleError::OwnerTemplateMismatch { .. }
            ))
        ));
    }

    #[test]
    fn ordinary_emission_rejects_malformed_bundle_parity() {
        use bloq_ir::{BundleDetector, BundleMeasurement, DetectorBundle, DetectorBundleUse};

        for term in [
            DetectorTerm::Measurement(BundleMeasurement {
                owner: 0,
                measurement: 99,
            }),
            DetectorTerm::LoopState(LoopStateId(7)),
        ] {
            let mut graph = Bloq::new();
            let mut circuit = CoordCircuit::new();
            circuit.measure(PauliBasis::Z, [IVec2::ZERO]);
            let mut node = template_node_with_circuit_at(
                &mut graph,
                circuit,
                ivec3(0, 0, 0),
                TemplateInstanceId(0),
                IVec2::ZERO,
            );
            let template = node.expect_quantum().instances[0].template_id;
            let bundle = graph.add_detector_bundle(DetectorBundle::new(
                vec![template],
                vec![BundleDetector {
                    parity: DetectorParity::from_terms([term]),
                    coords: None,
                }],
            ));
            node.expect_quantum_mut()
                .detector_bundles
                .push(DetectorBundleUse {
                    bundle,
                    instances: vec![TemplateInstanceId(0)],
                    offset: IVec2::ZERO,
                });
            graph.add_node(node);

            let error = emit_bloq_stim(&graph).expect_err("malformed annotation must fail");
            match term {
                DetectorTerm::Measurement(_) => assert_eq!(
                    error,
                    StimEmissionError::UnknownInstanceMeasurement(InstanceMeasurement {
                        instance: TemplateInstanceId(0),
                        measurement: 99,
                    }),
                ),
                DetectorTerm::LoopState(state) => {
                    assert_eq!(error, StimEmissionError::NodeLoopStateUnsupported(state),)
                }
            }
        }
    }

    #[test]
    fn emit_bloq_stim_maps_feedforward_control_to_its_instance_record() {
        use bloq_circuit::ConditionalCorrection;
        let mut graph = Bloq::new();
        let mut parent_circuit = CoordCircuit::new();
        parent_circuit.measure(PauliBasis::Z, [ivec2(0, 0)]);
        let parent_node = template_node_with_circuit_at(
            &mut graph,
            parent_circuit,
            ivec3(0, 0, 0),
            TemplateInstanceId(0),
            IVec2::ZERO,
        );

        let mut child_circuit = CoordCircuit::new();
        let control = child_circuit.measure(PauliBasis::Z, [ivec2(1, 0)])[0];
        child_circuit
            .body_mut(child_circuit.entry_body())
            .expect("entry body exists")
            .ops_mut()
            .push(Op::ConditionalPauli(vec![ConditionalCorrection {
                pauli: PauliBasis::X,
                control,
                target: ivec2(2, 0),
            }]));
        child_circuit
            .do_gate(bloq_circuit::GateType::RZ, [ivec2(2, 0)])
            .unwrap();
        let child_node = template_node_with_circuit_at(
            &mut graph,
            child_circuit,
            ivec3(0, 0, 1),
            TemplateInstanceId(1),
            IVec2::ZERO,
        );

        let parent = graph.add_node(parent_node);
        let child = graph.add_node(child_node);
        graph.add_edge(parent, child, BloqEdge::quantum(vec![]));

        for aligned in [false, true] {
            let text =
                emit_bloq_stim_with(&graph, &BloqStimOptions::new().with_align_moments(aligned))
                    .expect("emit measurement feed-forward");
            assert!(text.contains("M 1\nCX rec[-1] 2"), "{text}");
            assert!(!text.contains("CX rec[-2] 2"), "{text}");
            assert!(
                text.find("CX rec[-1] 2").unwrap() < text.find("R 2").unwrap(),
                "{text}"
            );
            #[cfg(feature = "verify")]
            text.parse::<stim::Circuit>()
                .expect("program feed-forward should parse as Stim");
        }
    }

    #[test]
    fn emit_bloq_stim_summarizes_nested_large_repeat_frames() {
        // A large inner repeat under a summarized outer one must not expand
        // per-iteration accounting: the frame only needs the final inner
        // iteration's occurrences, wherever the loop nests.
        let qubit = ivec2(0, 0);
        let mut circuit = CoordCircuit::new();
        let measurement = circuit.reserve_measurement_id(qubit);
        let inner = circuit.add_body(bloq_circuit::CircuitBody::from_ops(vec![Op::Measure {
            basis: PauliBasis::Z,
            qubits: vec![qubit],
            measurements: vec![measurement],
            flip_probability: 0.0,
        }]));
        let outer = circuit.add_body(bloq_circuit::CircuitBody::from_ops(vec![Op::Repeat {
            body: inner,
            repetitions: 1_000_000,
        }]));
        circuit.push_repeat(outer, 2);
        let mut graph = Bloq::new();
        let mut node = template_node_with_circuit(&mut graph, circuit);
        node.expect_quantum_mut().detectors.push(NodeDetector {
            parity: DetectorParity::from_measurements([InstanceMeasurement {
                instance: TemplateInstanceId(0),
                measurement,
            }]),
            coords: None,
        });
        graph.add_node(node);

        let text = emit_bloq_stim(&graph).expect("emit Bloq Stim");

        assert!(text.contains("REPEAT 2 {"), "{text}");
        assert!(text.contains("REPEAT 1000000 {"), "{text}");
        assert_eq!(
            text.lines().filter(|line| line.trim() == "M 0").count(),
            1,
            "{text}"
        );
        assert!(text.contains("DETECTOR rec[-1]"), "{text}");
    }
}
