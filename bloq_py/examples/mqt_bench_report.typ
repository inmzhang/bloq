#set page(paper: "a4", margin: (x: 18mm, y: 17mm), numbering: "1")
#set text(size: 9pt)
#set par(justify: true)
#set heading(numbering: "1.")
#show heading.where(level: 1): set text(size: 16pt, fill: rgb("183c52"))
#show heading.where(level: 2): set text(size: 12pt)
#show raw: set text(size: 6.5pt)
#show link: set text(fill: rgb("246385"))

#let data = json("report-summary.json")
#let assets = json("report-assets.json")
#let runs = data.runs
#let compiled(row) = row.at("ir_d25", default: (:)).at("status", default: "") == "compiled"
#let successes = runs.filter(compiled)
#let failures = runs.filter(row => not compiled(row))
#let wrapped(row) = row.at("source_variant", default: "") == "explicit_input_preparation"
#let primary = runs.filter(row => row.experiment == "default" and not wrapped(row))
#let supplements = runs.filter(row => row.experiment == "default" and wrapped(row))
#let additional = runs.filter(row => row.experiment != "default")
#let value(dict, key, default: "—") = dict.at(key, default: default)
#let shown(x) = if x == none { "—" } else if type(x) == str { x } else { json.encode(x, pretty: false) }
#let number(x) = if x == none { "—" } else { str(calc.round(x, digits: 5)) }
#let brief(message) = {
  let chars = message.clusters()
  if chars.len() > 600 { chars.slice(0, 600).join() + "… (full diagnostic in ledger and logs)" } else { message }
}
#let status(row, stage) = value(row.at(stage, default: (:)), "status", default: "skipped")
#let case-name(row) = row.source_case.replace("_", " ")
#let profile(row) = {
  let options = row.at("layout_options", default: (:))
  [spacing=#value(options, "spacing"), routing margin=#value(options, "routing_margin"),
    placement margin=#value(options, "placement_margin")]
}
#let introductions = (
  half_adder: "A reversible half adder computes a sum bit and carry using controlled operations.",
  full_adder: "A reversible full adder includes an incoming carry in the arithmetic mapping.",
  grover: "Grover search amplifies marked computational-basis states through oracle and diffusion steps.",
  qpeexact: "Phase estimation extracts an eigenphase using controlled unitary powers and an inverse Fourier transform.",
  qpeinexact: "Phase estimation samples a phase that is not exactly representable by the counting register.",
  qft: "The quantum Fourier transform maps computational-basis amplitudes into a discrete Fourier basis.",
  qftentangled: "The quantum Fourier transform is applied to MQT's entangled input preparation.",
  wstate: "W-state preparation creates a coherent superposition with exactly one excited qubit.",
  qwalk: "A quantum walk evolves a coherent position state using quantum-controlled transitions.",
  qaoa: "This MaxCut QAOA circuit alternates problem and mixing unitaries at fixed MQT parameters; it does not run a classical optimization loop.",
  vqe_real_amp: "This real-amplitude variational ansatz evaluates fixed MQT parameters; it does not run an energy-minimization loop.",
  vqe_two_local: "This variational ansatz alternates single-qubit rotations and entanglers at fixed MQT parameters; no optimization loop is executed.",
  qnn: "A quantum neural-network kernel combines a feature map and a parameterized ansatz at fixed inputs and weights.",
  bmw_quark_cardinality: "A QUARK generative-modeling circuit combines single-qubit rotations and XX interactions at fixed MQT parameters.",
  bmw_quark_copula: "A QUARK generative-modeling circuit starts with correlations between two registers and applies parameterized ansatz layers.",
  cdkm_ripple_carry_adder: "The CDKM reversible adder propagates carry information through an arithmetic register.",
  vbe_ripple_carry_adder: "The VBE reversible adder implements addition using explicit carry operations.",
  multiplier: "Reversible multiplication builds arithmetic products from controlled operations.",
  hrs_cumulative_multiplier: "The HRS cumulative multiplier accumulates a product in a reversible arithmetic register.",
  draper_qft_adder: "Draper addition performs reversible arithmetic through a Fourier-basis register.",
  modular_adder: "A reversible modular adder adds one register into another modulo two to the register width.",
  ae: "Amplitude estimation uses controlled Grover rotations and phase estimation to estimate a success probability.",
  hhl: "The HHL circuit implements the phase-estimation and controlled-rotation kernel for a small linear system.",
  bv: "Bernstein–Vazirani recovers a hidden bit string from a coherent oracle query.",
  dj: "Deutsch–Jozsa distinguishes a constant oracle from a balanced oracle.",
  graphstate: "Graph-state preparation entangles qubits according to a seeded graph.",
)
#let introduction(row) = introductions.at(row.at("benchmark", default: ""), default: value(row, "note"))
#let artifact-links(artifacts) = {
  let links = artifacts.pairs().map(((name, path)) => link(path, name))
  links.join([ · ])
}
#let evidence(row, stage) = {
  let check = row.at(stage, default: (:))
  [*#stage:* #status(row, stage)]
  if check.at("reason", default: none) != none { [; #brief(check.reason)] }
  if check.at("error", default: none) != none { [; #brief(check.error)] }
  if stage == "quizx" and check.at("samples", default: none) != none {
    [; #value(check, "verified") / #check.samples sampled branches verified]
  }
  if stage == "vm_d3" and check.at("shots", default: none) != none {
    let shots = check.shots
    [; #shots.len() sampled shots;
      minimum source-state fidelity #number(calc.min(..shots.map(shot => shot.source_state_fidelity)));
      maximum source Pauli error #shown(calc.max(..shots.map(shot => shot.source_max_pauli_error)));
      #value(check, "checked_paulis") joint Pauli expectations per shot]
  }
  parbreak()
}

#align(center)[
  #text(size: 23pt, weight: "bold", fill: rgb("183c52"))[MQT Bench → Bloq]
  #v(2mm)
  #text(size: 12pt)[Reproducible algorithm compilation experiments]
]

This catalog is generated from the #link("report-summary.json")[derived report ledger].
The original #link("summary.json")[default experiment ledger] remains unchanged.
MQT Bench supplies the algorithm circuits; the same lowered Clifford+T QASM is
used across upstream compilers. The catalog contains #primary.filter(compiled).len()
validated distance-25 compilations of unmodified MQT inputs and
#primary.filter(row => not compiled(row)).len() other attempted outcomes.
#if supplements.len() > 0 [A separate section contains
  #supplements.filter(compiled).len() successful supplementary compilations with
  explicit input preparation.]
#if additional.len() > 0 [Additional independently recorded experiments contain
  #additional.filter(compiled).len() validated compilations and
  #additional.filter(row => not compiled(row)).len() other outcomes; they retain
  their own configurations and provenance.]
Full provenance: #link("manifest.json")[environment and revisions],
#link("artifacts.json")[artifact hashes], and #link("report.typ")[report source].

= What the evidence establishes

Source circuits are exported as OpenQASM 3 and compiler inputs as OpenQASM 2.
Terminal measurement mappings are retained separately; unsupported dynamic
operations are rejected. Exact decompositions preserve gates up to global phase.
When matrix-valued gates cannot be exported directly, source QASM uses exact
definition expansion and SWAP permutation synthesis. The ledger records this
serialization step; numerical checks retain the untouched MQT circuit as their reference.
Other rotations use a stated circuit-wide synthesis budget. Dense numerical
operator comparisons test every input numerically; sampled statevector checks
cover only the listed inputs. Neither is a formal equivalence proof.
Supplementary coherent-input demonstrations prepend only the recorded input
preparation. The original MQT algorithm is retained unaltered as benchmark.qasm;
the prepared executable circuit is source.qasm. Those demonstrations are separated
from unmodified MQT benchmark inputs.

IR compilation at requested distance 25 uses ideal T-source proxies, full compiler validation,
and an exact binary serialization round trip. Physical execution, when present,
uses distance 3, ideal T proxies, noiseless physical circuits, and ideal boundary
capture. Output-state checks cover the prepared input and sampled adaptive
branches. QuiZX checks sample logical branches against the lowered circuit.
Compilation, sampled execution, and synthesis approximation are separate claims.
The VM starts from the all-zero input, including the MQT circuit's own
preparation gates. Passing those shots establishes that output state, not the
complete algorithm channel on arbitrary inputs.
Compiler warnings about reduced effective distance are retained per run; a
requested distance is not a measured circuit-distance guarantee.

Approximation bounds exclude floating-point evaluation error. A failed or
inconclusive semantic check remains visible even when IR compilation succeeds.
Large compilation-only cases do not establish complete algorithm equivalence.
Lowered T/T† gate counts and routed T-resource counts are reported separately.
Neither count alone proves a non-Clifford channel or a non-stabilizer output:
optimization can cancel T gates, and a non-Clifford channel can produce a
stabilizer state for the tested input. In particular, the small half adder,
Grover, exact QPE, and full adder can pass the VM while their all-zero-input
outputs remain stabilizer states. The three-qubit quantum walk simplifies to
zero routed T resources in SpiderLS and is not a magic-state demonstration.

= Results at a glance

#set table(inset: 3pt, stroke: .35pt + luma(190))
#let overview(rows) = table(
  columns: (1.8fr, 1fr, .45fr, .65fr, 1fr, 1fr),
  table.header([*Circuit*], [*Compiler*], [*Q*], [*Lowered / routed T*], [*d=25 IR*], [*Evidence*]),
  ..rows.map(row => (
    [#case-name(row)], [#value(row, "compiler", default: "source")],
    [#shown(value(row, "logical_qubits"))], [#shown(value(row, "t_count")) / #shown(value(row, "routed_t"))],
    [#status(row, "ir_d25")], [#value(row, "verification", default: value(row, "status"))],
  )).flatten(),
)

== Unmodified MQT benchmark inputs

#overview(primary)

#if supplements.len() > 0 [
  == Supplementary explicit-input demonstrations

  #overview(supplements)
]

#if additional.len() > 0 [
  == Additional routing experiments

  These separately recorded configurations are not replacements for the default
  results above. Each experiment retains its original ledger and provenance.
  #for session in data.experiments.filter(session => session.id != "default") [
    *#session.label:* #link(session.summary_file)[original ledger]

    #let rows = additional.filter(row => row.experiment == session.id)
    #for row in rows [#case-name(row) via #row.compiler: #profile(row). #parbreak()]
    #overview(rows)
  ]
]

== Paper-demo candidates

#let candidates = successes.filter(row => value(row, "routed_t", default: 0) > 0
  and row.at("output_stabilizer", default: true) == false and status(row, "vm_d3") == "sampled"
  and status(row, "quizx") == "sampled")
#if candidates.len() == 0 [
  No retained run currently has a numerically classified non-stabilizer output,
  routed T resources, sampled QuiZX verification, and a passing distance-3 VM
  check together. Other successful runs remain pipeline demonstrations;
  compilation-only results should not be presented as execution demonstrations.
] else [
  These runs have both sampled logical and physical evidence, in addition to
  validated distance-25 compilation, retained T resources, and numerically
  classified non-stabilizer source outputs. They demonstrate magic-state
  execution on the prepared input; the sampled evidence does not prove every branch.
  #for row in candidates [
    - *#case-name(row)* via #row.compiler: #row.logical_qubits qubits,
      #row.t_count lowered T/T† gates / #row.routed_t routed T resources;
      output stabilizer: #shown(value(row, "output_stabilizer", default: "not checked")),
      channel Clifford: #shown(value(row, "channel_clifford", default: "not checked")),
      input variant: #value(row, "source_variant", default: "unmodified_mqt_benchmark").
  ]
]

#let run-page(row) = {
  pagebreak(weak: true)
  heading(level: 2)[#case-name(row) via #row.compiler]
  [*Experiment:* #row.experiment_label.]
  parbreak()
  if row.experiment != "default" {
    [*Additional routing configuration:* #profile(row).]
    parbreak()
  }
  [#introduction(row) #value(row, "note").]
  parbreak()
  if wrapped(row) {
    [*Supplementary explicit input:* #shown(value(row, "input_preparation"));
      the original MQT algorithm is unchanged.]
    parbreak()
    link(assets.cases.at(row.case).at("benchmark.qasm"), [Original MQT benchmark QASM])
    parbreak()
  }
  let links = assets.runs.at(row.run_dir)
  artifact-links(links)
  parbreak()
  if "layout.svg" in links or "layout.png" in links {
    image(links.at("layout.png", default: links.at("layout.svg", default: "")),
      width: 100%, height: 85mm, fit: "contain")
  } else {
    [Static image unavailable within its rendering budget:
      #shown(assets.thumbnails.at(row.run_dir))]
    parbreak()
  }
  if "layout.html" in links {
    align(center)[#text(size: 8pt)[Static projection of the generated block graph;
      #link(links.at("layout.html"))[open the interactive 3D viewer].]]
  } else {
    [Interactive viewer unavailable within its rendering budget:
      #shown(assets.viewers.at(row.run_dir, default: (:))) ]
  }
  parbreak()
  let ir = row.ir_d25
  table(columns: (1fr, 1fr, 1fr, 1fr),
    [*Logical qubits*], [*Lowered / routed T*], [*Blocks / pipes*], [*Extents*],
    [#row.logical_qubits], [#row.t_count / #shown(value(row, "routed_t"))],
    [#value(row, "blocks", default: value(row, "layout_blocks")) / #value(row, "pipes")],
    [#shown(value(row, "extents"))],
    [*Physical-qubit IDs (d=25)*], [*IR nodes*], [*Layout seconds*], [*IR seconds*],
    [#value(ir, "physical_qubits")], [#value(ir, "nodes")],
    [#number(value(row, "layout_seconds", default: none))], [#number(value(ir, "seconds", default: none))],
  )
  parbreak()
  [*Lowering:* #value(row, "lowering_check"),
    error #shown(value(row, "lowering_error")),
    synthesis budget #shown(value(row, "approximation_bound")),
    exact lowering #shown(value(row, "exact_lowering")).]
  parbreak()
  [*Source-export error:* #shown(value(row, "source_export_error")),
    *serialization:* #value(row, "source_serialization", default: "original_gate_definitions"),
    *Hadamard pipes by axis:* #shown(value(row, "hadamard_pipes_by_axis")),
    *routed T resources:* #shown(value(row, "routed_t"))]
  parbreak()
  [*Zero-input source output stabilizer:* #shown(value(row, "output_stabilizer", default: "not checked")),
    *source channel Clifford:* #shown(value(row, "channel_clifford", default: "not checked")).]
  parbreak()
  [*d=25:* validated with ideal T proxies; binary round trip passed.]
  parbreak()
  for warning in ir.at("warnings", default: ()) {
    [*Compiler warning:* #warning]
    parbreak()
  }
  evidence(row, "quizx")
  evidence(row, "vm_d3")
  [*Classification:* #value(row, "verification")]
  parbreak()
}

= Successful unmodified-input compilations

#for row in primary.filter(compiled) { run-page(row) }

#if supplements.len() > 0 [
  = Supplementary explicit-input compilations

  #for row in supplements.filter(compiled) { run-page(row) }
]

#if additional.len() > 0 [
  = Successful additional routing experiments

  #for row in additional.filter(compiled) { run-page(row) }
]

= Other attempted outcomes

Every rejected, failed, timed-out, or incomplete attempt remains in the ledger.
These outcomes describe the attempted configuration and budget; they do not prove
that an algorithm has no valid surface-code layout.

#for row in failures {
  let stage = value(row, "failed_stage", default: value(row, "stage", default: "lowering"))
  let failure = row.at(stage, default: row)
  heading(level: 2)[#case-name(row) / #value(row, "compiler", default: "source")]
  [*Experiment:* #row.experiment_label.]
  parbreak()
  [*Outcome:* #value(failure, "status", default: value(row, "status"));
    *stage:* #stage;
    *layer:* #value(row, "failure_layer");
    *code:* #value(failure, "error_code", default: value(row, "error_code"))]
  parbreak()
  let diagnostic = value(failure, "error", default: value(failure, "reason", default: value(row, "error", default: value(row, "note"))))
  [#brief(diagnostic)]
  parbreak()
  if row.at("run_dir", default: none) != none {
    artifact-links(assets.runs.at(row.run_dir, default: (:)))
    parbreak()
    let process = row.at("processes", default: (:)).at(stage, default: (:))
    for name in ("stdout", "stderr") {
      if name in process {
        link(row.run_dir + "/" + process.at(name), [Compilation #name log])
        [ ]
      }
    }
    parbreak()
  } else {
    artifact-links(assets.cases.at(row.case, default: (:)))
    parbreak()
  }
}

= Source circuits and adaptations

Complete source QASM, lowered compiler QASM, and BLOG are linked from each run.
Short source circuits are also reproduced below. Sources over 160 lines and
expanded Clifford+T circuits are linked in full to keep this catalog usable.
Source files are shared across compilers; their hashes are recorded in the ledger.

#let cases = successes.map(row => row.case).sorted().dedup()
#for case in cases {
  let row = successes.find(row => row.case == case)
  let sources = assets.cases.at(case)
  heading(level: 2)[#case-name(row) / #row.experiment_label]
  [#introduction(row) #value(row, "note").]
  parbreak()
  [MQT benchmark: #value(row, "benchmark"); requested size #value(row, "circuit_size");
    parameter seed #value(row, "mqt_parameter_seed");
    source hash #text(size: 7pt)[#value(row, "source_sha256")].]
  parbreak()
  [*Input variant:* #value(row, "source_variant", default: "unmodified_mqt_benchmark"),
    *explicit preparation:* #shown(value(row, "input_preparation", default: ())) .]
  parbreak()
  if wrapped(row) {
    [The benchmark.qasm artifact preserves the original MQT algorithm;
      source.qasm adds the declared preparation before that algorithm.]
    parbreak()
  }
  [Terminal measurement mapping:
    #text(size: 7pt)[#shown(value(row, "terminal_measurements"))].]
  parbreak()
  artifact-links(sources)
  parbreak()
  let qasm = read(sources.at("source.qasm"))
  if qasm.split("\n").len() <= 160 {
    raw(qasm, block: true)
  } else {
    [Full source: #qasm.split("\n").len() lines; use the source.qasm link above.]
    parbreak()
  }
}

= Reproducibility

The machine-readable manifest records dependency versions, upstream revisions and
patches, Bloq revision and dirty-state provenance, seeds, synthesis budget, and
experiment limits. The artifact manifest fingerprints retained source, layout,
IR, logs, and this PDF. Verify the copied bundle with
#raw("python bloq_py/examples/mqt_bench_report.py <output-dir> --verify").
No historical report is used as evidence for this catalog.

#for path in assets.provenance {
  link(path, path)
  [ · ]
}

== Recorded manifest

#for session in data.experiments {
  heading(level: 2)[#session.label]
  link(session.summary_file, [Original experiment ledger])
  parbreak()
  raw(json.encode(session.manifest, pretty: true), block: true)
}
