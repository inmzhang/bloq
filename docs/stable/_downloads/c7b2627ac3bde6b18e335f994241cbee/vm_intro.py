"""Lower a CNOT once and run three independent VM shots."""

from pathlib import Path

import bloq

ir = bloq.compile(bloq.GalleryItem.CNOT.load(), distance=3)
program = bloq.lower_vm(ir, decoder_latency_rounds=10)
for seed in range(3):
    result = program.run(seed=seed, input_state="plus")
    assert not result.discarded
    print(seed, result.discarded, result.finished_at)

Path("cnot.instructions.json").write_text(program.to_json(indent=2), encoding="utf-8")
Path("cnot.trace.json").write_text(result.to_json(indent=2), encoding="utf-8")
