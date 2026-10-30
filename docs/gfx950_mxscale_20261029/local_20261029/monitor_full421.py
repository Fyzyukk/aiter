#!/usr/bin/env python3
"""Read live full421 progress; this helper does not import GPU libraries."""
import csv
import json
from pathlib import Path

local = Path(__file__).resolve().parent
run = local / "step4_full421_top8"
fields = ("gfx", "cu_num", "M", "N", "K", "w_scale_block")
key = lambda row: tuple(str(row[field]) for field in fields)
with (local.parent / "handoff_data/csv_baseline.csv").open() as stream:
    baseline = {key(row): float(row["us"]) for row in csv.DictReader(stream)}
events = {}
path = run / "progress.jsonl"
if path.exists():
    for line in path.read_text().splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        events[key(event["shape"])] = event
wins = 0
no_correct = 0
for shape, event in events.items():
    result = event.get("best_opus")
    correct = bool(result and 0 <= float(result.get("errRatio", 1)) <= .05
                   and float(result.get("us", 0)) > 0)
    no_correct += not correct
    wins += bool(correct and float(result["us"]) < baseline[shape])
snapshots = {}
for path in sorted((run / "candidates").glob("*.json")):
    try:
        record = json.loads(path.read_text())
    except json.JSONDecodeError:
        continue
    shape = tuple(record["info_keys"])
    previous = snapshots.get(shape)
    if previous is None or previous["updated_utc"] <= record["updated_utc"]:
        snapshots[shape] = record
current = []
for record in snapshots.values():
    if record["phase"] != "running":
        continue
    current.append({"gpu": record["gpu_id"], "MNK": record["info_keys"][2:5],
                    "done": record["processed_total"], "total": record["total_tasks"],
                    "valid": record["valid_total"], "backend": record["current_libtype"],
                    "updated": record["updated_utc"]})
exit_file = run / "exit_code"
print(json.dumps({"completed": len(events), "expected": 421, "wins_csv": wins,
                  "losses_csv": len(events)-wins, "no_correct_opus": no_correct,
                  "pending": 421-len(events),
                  "active_candidate_results": sum(row["done"] for row in current),
                  "active_correct_candidate_results": sum(row["valid"] for row in current),
                  "failed_worker_shapes": sum(row["phase"] == "failed" for row in snapshots.values()),
                  "current": sorted(current, key=lambda row: row["gpu"]),
                  "exit_code": int(exit_file.read_text()) if exit_file.exists() else None}))
