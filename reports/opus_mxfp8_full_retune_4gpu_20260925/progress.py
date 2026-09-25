"""Print compact live progress without importing GPU libraries."""
import csv
import json
from collections import Counter
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent
rows = []
for gpu in range(4, 8):
    p = HERE / f"gpu{gpu}_r3_run.json"
    if not p.exists():
        continue
    r = json.loads(p.read_text())
    with (HERE / f"gpu{gpu}_r3_choices.csv").open() as f:
        choices = list(csv.DictReader(f))
    with (HERE / f"gpu{gpu}_r3_correctness.csv").open() as f:
        failures = Counter(x["lib"] for x in csv.DictReader(f) if x["status"] != "passed")
    n = len(choices)
    rows += [dict(row, gpu=gpu) for row in choices]
    elapsed = (r.get("end_time", time.time()) - r["start_time"]) / 60
    print(json.dumps(dict(gpu=gpu, status=r["status"], completed=n, total=len(r["shapes"]),
                          elapsed_min=round(elapsed, 1), rejected_checks=dict(failures),
                          error=r.get("error"))))
losses = [r for r in rows if r["opus_faster"] != "True"]
print(json.dumps(dict(completed=len(rows), opus_wins=len(rows)-len(losses), remaining=len(losses),
                      remaining_by_N_K=dict(Counter(f"{r['N']}/{r['K']}" for r in losses)),
                      best_opus=dict(Counter(r["opus"] for r in rows)))))
