"""Build a CNOT, close a compatible experiment, and sample its Stim circuit."""

from pathlib import Path

import bloq

# [build-start]
graph = bloq.BlockGraph()
blocks = [
    ((0, 0, 0), "Port"), ((0, 0, 1), "ZXX"),
    ((0, 0, 2), "ZXZ"), ((0, 0, 3), "Port"),
    ((0, 1, 1), "ZXX"), ((0, 1, 2), "ZXZ"),
    ((1, 1, 0), "Port"), ((1, 1, 1), "ZXZ"),
    ((1, 1, 2), "ZXZ"), ((1, 1, 3), "Port"),
]
for position, kind in blocks:
    graph.add_block(bloq.Block(position, kind))
for position, direction in [
    ((0, 0, 0), "+Z"), ((0, 0, 1), "+Z"),
    ((0, 0, 1), "+Y"), ((0, 0, 2), "+Z"),
    ((0, 1, 1), "+Z"), ((0, 1, 2), "+X"),
    ((1, 1, 0), "+Z"), ((1, 1, 1), "+Z"),
    ((1, 1, 2), "+Z"),
]:
    graph.add_pipe(bloq.Pipe(position, direction))
graph.validate()
graph.save("cnot.blog")
# [build-end]

# [correlations-start]
correlations = graph.stabilizers()
assert len(correlations) == 4
for row in correlations:
    print(row.kind, row.stabilizer.port_stabilizer)
# [correlations-end]

# [compile-start]
program = bloq.compile(graph, distance=3)
program.validate()
program.save("cnot.bloqir")
Path("cnot.svg").write_text(program.to_svg(include_classical=True), encoding="utf-8")
circuit = bloq.emit_stim(program)
circuit.to_file("cnot.stim")
# [compile-end]

# [fill-start]
experiments = graph.fill_ports_auto()
closed, measured_relations = experiments[0]
assert closed.port_count == 0
closed.save("cnot.closed.blog")
closed_program = bloq.compile(closed, distance=3)
clean = bloq.emit_stim(closed_program)
detectors, observables = clean.compile_detector_sampler(seed=7).sample(
    shots=1024, separate_observables=True
)
assert not detectors.any()
assert not observables.any()
# [fill-end]

# [noise-start]
import numpy as np
import pymatching

noisy = bloq.emit_stim(closed_program, noise=1e-3)
noisy.to_file("cnot.closed.stim")
model = noisy.detector_error_model(decompose_errors=True)
matching = pymatching.Matching.from_detector_error_model(model)
detectors, actual = noisy.compile_detector_sampler(seed=11).sample(
    shots=1024, separate_observables=True
)
predicted = matching.decode_batch(detectors)
failures = np.any(predicted != actual, axis=1)
print("noisy shots:", len(failures))
print("any-logical failures:", int(failures.sum()))
# [noise-end]
