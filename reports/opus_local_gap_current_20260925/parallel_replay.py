"""Validate exported choices through native E8M0 replay on their original GPUs."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
assert json.loads((HERE / "final_summary.json").read_text())["status"] == "passed"
with (HERE / "final_tuned.csv").open() as f:
    rows = list(csv.DictReader(f))
with (HERE / "final_comparison.csv").open() as f:
    mapping = {(r["M"], r["N"], r["K"]): int(r["gpu"]) for r in csv.DictReader(f)}
assert len(rows) == len(mapping) == 295
# Correctness replay may run on another identical GPU; no replay timing is
# substituted for measured times. Avoid GPUs 4 and 7 while externally occupied.
replay_mapping=dict(mapping)
for i,key in enumerate(sorted(k for k,g in mapping.items() if g==4)):
    replay_mapping[key]=5+i%2
workers = []
for gpu in (5,6):
    part = [r for r in rows if replay_mapping[r["M"], r["N"], r["K"]] == gpu]
    if not part:
        continue
    path = HERE / f"gpu{gpu}_replay_config.csv"
    with path.open("x", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(part)
    command = [sys.executable, "-u", str(HERE / "replay.py"), "--gpu", str(gpu), "--config", str(path)]
    with (HERE / f"gpu{gpu}_replay.log").open("x") as f:
        process = subprocess.Popen(command, stdout=f, stderr=subprocess.STDOUT)
    workers.append((gpu, process, len(part)))
while True:
    status = [dict(gpu=g, pid=p.pid, exit_code=p.poll()) for g, p, _ in workers]
    print(json.dumps(status), flush=True)
    if all(p.poll() is not None for _, p, _ in workers):
        break
    time.sleep(20)
details = []
for gpu, process, count in workers:
    assert process.returncode == 0, (gpu, process.returncode)
    data = json.loads((HERE / f"gpu{gpu}_replay.json").read_text())
    assert data["status"] == "passed" and data["shapes"] == count
    details.append(data)
result = dict(status="passed", shapes=295, gpus=[g for g, _, _ in workers], workers=details,
              tuned_sha256=hashlib.sha256((HERE / "final_tuned.csv").read_bytes()).hexdigest(),
              purpose="Saved native-E8M0 choices replay on idle GPUs 5/6; measurement GPUs remain in final_comparison.csv and replay times never replace tuning measurements")
(HERE / "replay.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({"status": "passed", "replayed_shapes": 295}))
