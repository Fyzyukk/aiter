"""Run independent shape shards on physical GPUs 4--7, bound by UUID."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
plan = json.loads((HERE / "plan.json").read_text())
state = dict(status="running", start_time=time.time(), pid=os.getpid(), workers=[])
processes = []


def save():
    state["updated_time"] = time.time()
    tmp = HERE / "launcher.json.tmp"
    tmp.write_text(json.dumps(state, indent=2) + "\n")
    tmp.replace(HERE / "launcher.json")


try:
    assert not (HERE / "launcher.json").exists(), "Use a fresh run directory"
    for worker in plan["workers"]:
        env = os.environ.copy()
        for key in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
            env.pop(key, None)
        env["OPUS_TUNE_GPU_INDEX"] = str(worker["gpu"])
        command = [sys.executable, "-u", str(HERE / "benchmark.py"), "--sweep",
                   "--prefix", worker["prefix"], "--rounds", str(plan["rounds"]),
                   "--shapes", worker["shapes_file"], "--jit-dir", plan["jit_dir"]]
        with Path(worker["prefix"] + ".log").open("x") as log:
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        processes.append(process)
        state["workers"].append(dict(worker, pid=process.pid, status="starting"))
    save()
    while True:
        for process, worker in zip(processes, state["workers"]):
            path = Path(worker["prefix"] + "_run.json")
            if path.exists():
                manifest = json.loads(path.read_text())
                for key in ("status", "completed_shapes", "timing_records", "error"):
                    if key in manifest:
                        worker[key] = manifest[key]
            worker["exit_code"] = process.poll()
        save()
        print(json.dumps({"workers": [{k: w.get(k) for k in
              ("gpu", "status", "completed_shapes", "exit_code")} for w in state["workers"]]}), flush=True)
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
