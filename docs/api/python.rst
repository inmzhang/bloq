Python API Reference
====================

The ``bloq`` namespace provides the main workflow: author or load a
``BlockGraph``, compile it to ``Bloq``, then emit a circuit or lower the program
for VM verification. Import ``bloq.ir`` for the concrete types returned by IR
accessors. These are the same native objects, with no conversion layer.

This page groups signatures and generated docstrings by task. For a complete
first program, use the :doc:`quickstart <../getting-started/quickstart>`.

.. currentmodule:: bloq

.. automodule:: bloq.ir

Compilation
-----------

``compile`` returns a ``Bloq`` program at an odd code distance in ``3..=255``.
It accepts both local and hierarchical sources through ``BlockGraph``. Pass
``validate=True`` to request a full IR audit after compilation; required local
checks run in either case.

The free ``compile`` function shares a process-wide template cache with no
automatic eviction. Use ``clear_compile_cache`` between large distance sweeps,
or a ``CompileContext`` when a Python object should own a fixed configuration
and private cache. A context can serve concurrent callers. Budget overrides use
``limits={"FIELD": count}``; ``None`` disables that field's limit.

.. autofunction:: bloq.compile

.. autoclass:: bloq.CompileContext
   :members:
   :undoc-members:

.. autofunction:: bloq.clear_compile_cache

.. autofunction:: bloq.is_valid_distance

Clifford proxies
----------------

Clifford proxies replace T resources with perfect input ports and pin selective
sites to measurement arms. Supply the pins explicitly, or use the random helper
with a seed for reproducibility. These programs support circuit-distance
analysis; they do not verify the original non-Clifford computation or replace
structural branch selection.

.. autofunction:: bloq.compile_clifford_proxy

.. autofunction:: bloq.compile_random_clifford_proxy

Source authoring
----------------

``BlockGraph`` owns source geometry, actions, and module definitions. Its BLOG
load/save operations retain authored hierarchy; compilation consumes that same
graph representation. Call ``flatten()`` only when an independent flat projection
is needed, such as for an explicit port-filling experiment.

Use ``ActionDag`` and expressions for named measurements, feedback, and control.
``GalleryItem.load()`` supplies reusable examples. The
:doc:`BLOG format <../graphs/blog>` describes the corresponding source syntax.

.. autoclass:: bloq.BlockGraph
   :members:
   :undoc-members:

.. autoclass:: bloq.Block
   :members:
   :undoc-members:

.. autoclass:: bloq.BlockKind
   :members:
   :undoc-members:

.. autoclass:: bloq.Pipe
   :members:
   :undoc-members:

.. autoclass:: bloq.Action
   :members:
   :undoc-members:

.. autoclass:: bloq.ActionDag
   :members:
   :undoc-members:

.. autoclass:: bloq.Branch
   :members:
   :undoc-members:

.. autoclass:: bloq.BranchArm
   :members:
   :undoc-members:

.. autoclass:: bloq.MeasureTarget
   :members:
   :undoc-members:

.. autoclass:: bloq.FeedbackTarget
   :members:
   :undoc-members:

.. autoclass:: bloq.Expr
   :members:
   :undoc-members:

.. autoclass:: bloq.GalleryItem
   :members:
   :undoc-members:

IR programs and graph traversal
-------------------------------

``Bloq`` owns the physical program, shared templates, and nested control regions.
Loading a text or binary artifact decodes it without a full audit; call
``validate()`` when that audit is required. Structural statistics count stored
region bodies once, independently of how many attempts execution may take.

Node IDs are local to one graph level. Preserve the owning path in walk results
when identifying nested nodes; the same numeric ID can occur in different
regions. Accessors return snapshots, including detached ``SubGraph`` values, so
changing an inspected object does not edit the original program. Use the
program's editing methods for mutations.

.. autoclass:: bloq.Bloq
   :members:
   :undoc-members:

.. autoclass:: bloq.ir.BloqStats
   :members:
   :undoc-members:

.. autoclass:: bloq.ir.SubGraph
   :members:
   :undoc-members:

.. autoclass:: bloq.ir.BloqNode
   :members:
   :undoc-members:

.. autoclass:: bloq.ir.BloqNodeKind
   :members:
   :undoc-members:

.. autoclass:: bloq.ir.BloqEdge
   :members:
   :undoc-members:

.. autoclass:: bloq.ir.BloqEdgeRef
   :members:
   :undoc-members:

.. autoclass:: bloq.ir.WalkNode
   :members:
   :undoc-members:

.. autoclass:: bloq.ir.NodeKey
   :members:
   :undoc-members:

Quantum circuits and templates
------------------------------

A ``Template`` stores a reusable physical circuit; a ``TemplateInstance`` gives
it a program-wide identity and layout offset. A ``QuantumNode`` groups placed
instances, detectors, and restart syndromes. Template-local coordinates and
measurement indices acquire their instance meaning only after placement.

``QuantumGuard`` selects conditional membership. ``TemporalPipe`` and
``PipeSeam`` describe the quantum connection between stages, including recorded
memory padding. Custom backends must account for guards and seams as well as the
operations in each template.

.. autoclass:: bloq.ir.Circuit
   :members:
   :undoc-members:

.. autoclass:: bloq.ir.CircuitOp
   :members:
   :undoc-members:

.. autoclass:: bloq.ir.ConditionalCorrection
   :members:
   :undoc-members:

.. autoclass:: bloq.ir.ExpandedMeasurementColumns
   :members:
   :undoc-members:

.. autoclass:: bloq.ir.Template
   :members:
   :undoc-members:

.. autoclass:: bloq.ir.TemplateInstance
   :members:
   :undoc-members:

.. autoclass:: bloq.ir.QuantumNode
   :members:
   :undoc-members:

.. autoclass:: bloq.ir.QuantumGuard
   :members:
   :undoc-members:

.. autoclass:: bloq.ir.TemporalPipe
   :members:
   :undoc-members:

.. autoclass:: bloq.ir.PipeSeam
   :members:
   :undoc-members:

.. autoclass:: bloq.ir.PipePadding
   :members:
   :undoc-members:

.. autoclass:: bloq.ir.MemoryRoundTarget
   :members:
   :undoc-members:

Classical control and readouts
------------------------------

Value edges connect a producer's result to a consumer's input slot. A
``ClassicalExpr`` refers to those slots, not directly to arbitrary graph node IDs.
Keep operand order and parity constants when interpreting expressions.

Regions contain nested graphs and export values across their boundaries. A
retry region exports the accepted attempt's results. Activation decides whether
a node runs; it is separate from the expression's parity inputs. ``ValueRole``
records how dependencies participate in corrected readouts and feedback.

.. autoclass:: bloq.ir.ClassicalNode
   :members:
   :undoc-members:

.. autoclass:: bloq.ir.ClassicalExpr
   :members:
   :undoc-members:

.. autoclass:: bloq.ir.ClassicalResolution
   :members:
   :undoc-members:

.. autoclass:: bloq.ir.ValueInput
   :members:
   :undoc-members:

.. autoclass:: bloq.ir.ObservableOutput
   :members:
   :undoc-members:

.. autoclass:: bloq.ir.ValueRef
   :members:
   :undoc-members:

.. autoclass:: bloq.ir.ValueRole
   :members:
   :undoc-members:

.. autoclass:: bloq.ir.RegionNode
   :members:
   :undoc-members:

.. autoclass:: bloq.ir.RegionKind
   :members:
   :undoc-members:

.. autoclass:: bloq.ir.RegionRef
   :members:
   :undoc-members:

.. autoclass:: bloq.ir.InstanceMeasurement
   :members:
   :undoc-members:

.. autoclass:: bloq.ir.FramePair
   :members:
   :undoc-members:

Detectors and restart syndromes
-------------------------------

Detector and restart parities reference measurements in template-local or
placed-instance coordinates. A parity can also contain a constant sign or
loop-carried state, so a list of measurement indices alone may be incomplete.

Quantum nodes contain both inline detector rows and shared ``DetectorBundleUse``
entries. Each use binds a bundle's owner slots to instances and adds its placement
offset. Include both representations when counting or emitting detectors; use
activation applies to the whole bundle use.

.. autoclass:: bloq.ir.DetectorTerm
   :members:
   :undoc-members:

.. autoclass:: bloq.ir.TemplateParity
   :members:
   :undoc-members:

.. autoclass:: bloq.ir.TemplateDetector
   :members:
   :undoc-members:

.. autoclass:: bloq.ir.TemplateRestart
   :members:
   :undoc-members:

.. autoclass:: bloq.ir.TemplateRepeatState
   :members:
   :undoc-members:

.. autoclass:: bloq.ir.NodeParity
   :members:
   :undoc-members:

.. autoclass:: bloq.ir.NodeDetector
   :members:
   :undoc-members:

.. autoclass:: bloq.ir.NodeRestart
   :members:
   :undoc-members:

.. autoclass:: bloq.ir.DetectorBundle
   :members:
   :undoc-members:

.. autoclass:: bloq.ir.DetectorBundleUse
   :members:
   :undoc-members:

.. autoclass:: bloq.ir.BundleDetector
   :members:
   :undoc-members:

.. autoclass:: bloq.ir.BundleDetectorTerm
   :members:
   :undoc-members:

Logical boundaries and provenance
---------------------------------

Boundary operators bind logical Pauli support to a particular instance and its
input or output face. A ``LogicalOutput`` retains its owning instance: equal
qubit support does not make two physical cuts interchangeable.

Node and instance provenance link compiled work to source blocks, actions, and
module placements. Use that provenance for source-aware inspection, and allow
for synthetic nodes that have no authored origin.

.. autoclass:: bloq.ir.LogicalOutput
   :members:
   :undoc-members:

.. autoclass:: bloq.ir.InstanceBoundaryOperator
   :members:
   :undoc-members:

.. autoclass:: bloq.ir.BoundaryFace
   :members:
   :undoc-members:

.. autoclass:: bloq.ir.BoundaryFlow
   :members:
   :undoc-members:

.. autoclass:: bloq.ir.NodeProvenance
   :members:
   :undoc-members:

.. autoclass:: bloq.ir.InstanceProvenance
   :members:
   :undoc-members:

Stim backend
------------

Use ``compile_to_stim`` for a one-call source-to-circuit workflow, or
``emit_stim`` to reuse an existing IR program. Both return a ``stim.Circuit``;
use its ``to_file(path)`` method to save it. Stim emission requires a supported
fixed execution structure; retry regions and unpinned guarded quantum membership
need dynamic execution or an explicit supported projection first.

Emission plans and per-node segments expose the lowering for custom consumers.
Select ``dialect="stim"`` or ``dialect="clifft"`` where a text-emission API accepts
it. Isolated T-attempt artifacts expose cultivation circuits separately from the
whole program. See :doc:`emission <../backends/emission>` for backend restrictions.

.. autofunction:: bloq.compile_to_stim

.. autofunction:: bloq.emit_stim

.. autoclass:: bloq.ir.EmissionPlan
   :members:
   :undoc-members:

.. autofunction:: bloq.emit_plan_stim

.. autoclass:: bloq.PlanStim
   :members:
   :undoc-members:

.. autofunction:: bloq.emit_stim_segments

.. autofunction:: bloq.emit_stim_segments_pair

.. autoclass:: bloq.StimSegment
   :members:
   :undoc-members:

.. autoclass:: bloq.StimSegments
   :members:
   :undoc-members:

.. autofunction:: bloq.remap_stim_circuit

.. autofunction:: bloq.stim_to_clifft_text

.. autofunction:: bloq.clifft_to_stim_text

.. autofunction:: bloq.emit_isolated_t_attempts

.. autoclass:: bloq.IsolatedTAttemptArtifacts
   :members:
   :undoc-members:

.. autoclass:: bloq.IsolatedTAttemptManifest
   :members:
   :undoc-members:

VM verification
---------------

``run_noiseless`` executes native physical circuits and retry regions with
ideal input preparation and output-boundary capture. It returns corrected
logical marginals and optional joint Pauli signatures, including outputs whose
physical patches are reused later. It has no hardware timing or noise model.
Explicit noise already present in the supplied IR still executes.

.. autofunction:: bloq.run_noiseless

``lower_vm`` prepares a reusable ``VmProgram``; each ``run`` executes a dynamic
shot with its own seed and runtime limits. Check ``VmRunResult.discarded`` before
requesting corrected logical Bloch vectors, because a discarded shot has no
committed logical result.

The VM verifies measurement-dependent control and physical output states using
mock decoding and timing models. Its traces explain execution, but are not a
stable program persistence format or a claim about decoder performance. See
:doc:`VM verification <../backends/vm>` for model assumptions.

.. autofunction:: bloq.lower_vm

.. autoclass:: bloq.VmProgram
   :members:
   :undoc-members:

.. autoclass:: bloq.VmRunResult
   :members:
   :undoc-members:

ZX logical analysis
-------------------

Convert a source graph to its ZX diagram and inspect logical stabilizers before
physical compilation. This layer describes the source's logical relation;
physical detectors, scheduling, and decoder behavior require the compiled IR and
its backends. See :doc:`logical correlations <../theory/correlation-surfaces>`
for the relation between stabilizers and named readouts.

.. autofunction:: bloq.to_zx_graph

.. autoclass:: bloq.ZXGraph
   :members:
   :undoc-members:

.. autoclass:: bloq.ZXNode
   :members:
   :undoc-members:

.. autoclass:: bloq.ZXEdge
   :members:
   :undoc-members:

.. autoclass:: bloq.Stabilizer
   :members:
   :undoc-members:

.. autoclass:: bloq.StabilizerGenerator
   :members:
   :undoc-members:

Pauli operators and geometry
----------------------------

Pauli operators and measurement bases describe logical and physical support.
``PauliString`` represents Hermitian Pauli support without an explicit phase;
retain signs separately when an enclosing readout or parity carries them.
``Direction`` identifies a signed lattice direction, while ``UDirection``
identifies an unsigned axis.

.. autoclass:: bloq.Pauli
   :members:
   :undoc-members:

.. autoclass:: bloq.PauliBasis
   :members:
   :undoc-members:

.. autoclass:: bloq.PauliString
   :members:
   :undoc-members:

.. autoclass:: bloq.Basis
   :members:
   :undoc-members:

.. autoclass:: bloq.Direction
   :members:
   :undoc-members:

.. autoclass:: bloq.UDirection
   :members:
   :undoc-members:

Errors and warnings
-------------------

Catch ``BloqError`` for domain failures, or a specific subclass when recovery
depends on the stage: parsing, compilation, IR validation, emission, or VM
execution. A resource-limited ``CompileError`` means computation stopped at a
budget; it does not prove that the source is invalid.

Normal Python errors retain their meaning. Sequence indexing and mapping lookup
raise ``IndexError`` and ``KeyError``; argument coercion can raise ``TypeError``
or ``OverflowError``; filesystem access raises ``OSError``. ``CompileWarning``
uses Python's warnings system for advisories from successful compilation.

.. autoclass:: bloq.BloqError
   :members:
   :undoc-members:

.. autoclass:: bloq.InvalidArgumentError
   :members:
   :undoc-members:

.. autoclass:: bloq.ParseError
   :members:
   :undoc-members:

.. autoclass:: bloq.BlockGraphError
   :members:
   :undoc-members:

.. autoclass:: bloq.CompileError
   :members:
   :undoc-members:

.. autoclass:: bloq.CompileWarning
   :members:
   :undoc-members:

.. autoclass:: bloq.BloqValidationError
   :members:
   :undoc-members:

.. autoclass:: bloq.TextParseError
   :members:
   :undoc-members:

.. autoclass:: bloq.BinaryDecodeError
   :members:
   :undoc-members:

.. autoclass:: bloq.StimEmissionError
   :members:
   :undoc-members:

.. autoclass:: bloq.LowerError
   :members:
   :undoc-members:

.. autoclass:: bloq.RuntimeError
   :members:
   :undoc-members:

Format and metadata constants
-----------------------------

Use these constants for codec extensions, version checks, and compiler metadata
lookups instead of duplicating their string values. Exchange-format versions
identify the serialized layout; ``__version__`` identifies the installed Python
package. See :doc:`Bloq IR <../backends/ir>` for persistence choices.

.. autodata:: bloq.ir.BLOQ_TEXT_EXTENSION

.. autodata:: bloq.ir.BLOQ_TEXT_VERSION

.. autodata:: bloq.ir.BLOQ_BINARY_EXTENSION

.. autodata:: bloq.ir.BLOQ_BINARY_VERSION

.. autodata:: bloq.ir.CODE_DISTANCE_METADATA_KEY

.. autodata:: bloq.ir.CONVENTION_METADATA_KEY

.. autodata:: bloq.ir.CLIFFORD_PROXY_SEED_METADATA_KEY

.. autodata:: bloq.HONEST_T_TAG

.. autodata:: bloq.__version__
