"""Run preflight, all four shards, then audit/confirmation/export/replay."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
state = dict(status="running", stage="waiting_for_build", start_time=time.time())
path = HERE / "workflow.json"
assert not path.exists()
def save():
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(state, indent=2) + "\n")
    temp.replace(path)
save()
try:
    while True:
        build = json.loads((HERE / "build.json").read_text())
        if build["status"] != "running":
            assert build["status"] == "passed", build
            break
        time.sleep(10)
    state["stage"] = "smoke"
    save()
    smoke = HERE / "smoke_shapes.csv"
    with smoke.open("x", newline="") as f:
        w = csv.writer(f)
        w.writerow(("M", "N", "K"))
        w.writerows(((1088,6144,7168),(4096,2048,7168),(1536,7168,384)))
    env = dict(os.environ, OPUS_TUNE_GPU_INDEX="7")
    command = [sys.executable, "-u", str(HERE / "benchmark.py"), "--sweep",
               "--prefix", "smoke", "--rounds", "1", "--shapes", str(smoke),
               "--jit-dir", str(HERE / "jit")]
    with (HERE / "smoke.log").open("x") as log:
        r = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
    assert r.returncode == 0, "smoke failed"
    with (HERE / "smoke_correctness.csv").open() as f:
        errors = [r for r in csv.DictReader(f) if r["lib"] == "opus" and r["status"] != "passed"]
    assert not errors, errors
    state["stage"] = "full_sweep"
    save()
    with (HERE / "launch.log").open("x") as log:
        r = subprocess.run([sys.executable, "-u", str(HERE / "launch.py")], stdout=log, stderr=subprocess.STDOUT)
    assert r.returncode == 0, "full sweep failed"
    state["stage"] = "audit_confirmation_replay"
    save()
    with (HERE / "finish.log").open("x") as log:
        r = subprocess.run([sys.executable, "-u", str(HERE / "finish.py")], stdout=log, stderr=subprocess.STDOUT)
    assert r.returncode == 0, "finish pipeline failed"
    state.update(status="passed", stage="complete")
except BaseException as exc:
    state.update(status="failed", error=repr(exc))
    raise
finally:
    state["end_time"] = time.time()
    save()
