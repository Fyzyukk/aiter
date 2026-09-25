"""Confirm close or round-inconsistent shapes on their original physical GPUs."""
import csv
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
summary = json.loads((HERE / "summary.json").read_text())
assert summary["status"] == "passed"
plan = json.loads((HERE / "plan.json").read_text())
with (HERE / "close_shapes.csv").open() as f:
    rows = list(csv.DictReader(f))
shape = lambda r: tuple(int(r[k]) for k in ("M", "N", "K"))
state = dict(status="running", start_time=time.time(), pid=os.getpid(), shapes=len(rows),
             rounds=5, selection="Sweep absolute margin <=3% or inconsistent round winners", workers=[])
processes = []


def save():
    path = HERE / "confirmation.json.tmp"
    path.write_text(json.dumps(state, indent=2) + "\n")
    path.replace(HERE / "confirmation.json")


try:
    assert not (HERE / "confirmation.json").exists()
    for worker in plan["workers"]:
        with Path(worker["shapes_file"]).open() as f:
            own = {shape(r) for r in csv.DictReader(f)}
        subset = [r for r in rows if shape(r) in own]
        if not subset:
            continue
        gpu = worker["gpu"]
        path = HERE / f"gpu{gpu}_confirm_shapes.csv"
        with path.open("x", newline="") as f:
            w = csv.DictWriter(f, list(subset[0]))
            w.writeheader()
            w.writerows(subset)
        prefix = str(HERE / f"gpu{gpu}_confirm_r5")
        env = os.environ.copy()
        env["OPUS_TUNE_GPU_INDEX"] = str(gpu)
        command = [sys.executable, "-u", str(HERE / "benchmark.py"), "--final",
                   worker["prefix"] + "_run.json", "--prefix", prefix, "--rounds", "5",
                   "--shapes", str(path), "--jit-dir", plan["jit_dir"]]
        with Path(prefix + ".log").open("x") as log:
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        processes.append(process)
        state["workers"].append(dict(gpu=gpu, shapes=len(subset), prefix=prefix, pid=process.pid))
    save()
    while True:
        for worker, process in zip(state["workers"], processes):
            path = Path(worker["prefix"] + "_run.json")
            if path.exists():
                run = json.loads(path.read_text())
                for key in ("status", "completed_shapes", "error"):
                    if key in run:
                        worker[key] = run[key]
            worker["exit_code"] = process.poll()
        save()
        print(json.dumps(state["workers"]), flush=True)
        if all(p.poll() is not None for p in processes):
            break
        time.sleep(20)
    state["status"] = "passed" if all(w["exit_code"] == 0 and w["status"] == "passed"
                                      for w in state["workers"]) else "failed"
except BaseException as exc:
    state.update(status="interrupted", error=repr(exc))
    for process in processes:
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
    for process in processes:
        process.wait()
    raise
finally:
    state["end_time"] = time.time()
    save()
sys.exit(0 if state["status"] == "passed" else 1)
