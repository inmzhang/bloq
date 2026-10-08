"""Run THTH with causal timing; export its instructions and execution trace."""

import csv
import json
import math
from collections import Counter
from pathlib import Path

import bloq

# [example-start]
ir = bloq.compile(bloq.GalleryItem.THTH.load(), distance=3)
program = bloq.lower_vm(
    ir,
    gate_duration=1.0,
    decoder_latency_rounds=10,
    factory_release_time=0.0,
    source_release_time=23.0,
    input_release_time=23.0,
)
result = program.run(
    seed=17,
    input_state="plus",
    decoder_acceptance=1.0,
    accepted_accuracy=1.0,
    rejected_accuracy=1.0,
)
assert not result.discarded
expected = (math.sqrt(0.5), 0.5, 0.5)
assert all(math.isclose(a, b, abs_tol=1e-9) for a, b in zip(result.logical_bloch(), expected))
trace = result.trace
assert trace["metadata"]["decoder_latency_rounds"] == 10
assert all(span["end"] <= result.finished_at + 1e-9 for span in trace["timing"])
Path("thth.instructions.json").write_text(program.to_json(indent=2), encoding="utf-8")
Path("thth.trace.json").write_text(result.to_json(indent=2), encoding="utf-8")
print("logical Bloch vector:", result.logical_bloch())
print("finished at:", result.finished_at)
print("source roles:", dict(Counter(source["role"] for source in program.sources)))
# [example-end]

# Preserve separate physical and memory events. Task spans include nested work
# and therefore must not be summed to estimate hardware occupation.
tasks = {task["task"]: task for task in trace["metadata"]["tasks"]}
with Path("thth.timeline.csv").open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(
        file, fieldnames=["task", "function", "kind", "start", "end", "attempt", "wait_reason"]
    )
    writer.writeheader()
    for event in trace["events"]:
        if event["kind"] not in {"moment_issued", "memory_round", "idle"}:
            continue
        writer.writerow(
            {
                "task": event["task"],
                "function": tasks[event["task"]]["origin"]["function"],
                "kind": event["kind"],
                "start": event["start"],
                "end": event["end"],
                "attempt": event.get("attempt"),
                "wait_reason": event.get("wait_reason"),
            }
        )

# [adapter-start]
# Illustrative controller envelope, not a hardware driver. Retain the complete
# ISA: selecting alternatives and interpreting waits requires a live runtime.
instructions = json.loads(program.to_json())
controller_plan = {
    "time_unit": "one physical moment",
    "program": instructions,
    "logical_sites": {
        str(task["task"]): task["origin"]["sites"]
        for task in trace["metadata"]["tasks"]
        if task["origin"] is not None
    },
}
Path("thth.controller-plan.json").write_text(
    json.dumps(controller_plan, indent=2), encoding="utf-8"
)
assert controller_plan["program"]["tasks"] == instructions["tasks"]
# [adapter-end]
