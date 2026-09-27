#!/usr/bin/env python3
"""Shard whole target shapes across GPUs; keep finalists on their full-sweep GPU."""

import argparse
import csv
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

import benchmark as bench
import preflight


HERE = Path(__file__).resolve().parent
ROOT = bench.ROOT


def plan(shapes, gpus, mode, full_batch, batch_dir):
    assignments, full_inputs = {}, {}
    if mode == "full":
        assignments = {gpu: [] for gpu in gpus}
        cost = dict.fromkeys(gpus, 0.0)
        def weight(shape):
            m, n, k = shape
            return 1 + (m * k + n * k + 2 * m * n) / (64 * 1024**2)
        for shape in sorted(shapes, key=lambda s: (-weight(s), s)):
            gpu = min(gpus, key=lambda card: (cost[card], len(assignments[card]), card))
            assignments[gpu].append(shape)
            cost[gpu] += weight(shape)
    else:
        source = json.loads((full_batch / "launch.json").read_text())
        if source.get("external_mode") != "full":
            raise ValueError("--full-batch must refer to a full-mode batch")
        requested = set(shapes)
        for worker in source["workers"]:
            local = requested & {tuple(s) for s in worker["shapes"]}
            if not local:
                continue
            gpu = worker["gpu"]
            if gpu not in gpus or worker.get("status") != "passed" or worker.get("exit_code") != 0:
                raise ValueError(f"Requested shapes require successful full mode on GPU {gpu}")
            path = Path(worker["run_manifest"])
            if bench.base.sha256(path) != worker["run_sha256"]:
                raise ValueError("Full-mode worker manifest changed")
            assignments[gpu] = sorted(local)
            full_inputs[gpu] = str(path)
    covered = [shape for local in assignments.values() for shape in local]
    if len(covered) != len(set(covered)) or set(covered) != set(shapes):
        raise ValueError("GPU shards must cover every requested shape exactly once")
    return [dict(gpu=gpu, gpu_uuid=bench.GPU_MAP[gpu][0],
                 shapes=sorted(local, key=lambda s: (s[1], s[2], s[0])), shape_count=len(local),
                 shapes_file=str(batch_dir / f"gpu{gpu}_shapes.csv"),
                 prefix=str(batch_dir / f"gpu{gpu}"), full_manifest=full_inputs.get(gpu), status="prepared")
            for gpu, local in sorted(assignments.items()) if local]


def refresh(worker, process):
    path = Path(worker["prefix"] + "_run.json")
    if path.is_file():
        try:
            run = json.loads(path.read_text())
        except json.JSONDecodeError:
            run = {}
        for key in ("status", "completed_shapes", "timing_records", "rejected_checks", "error"):
            if key in run:
                worker[key] = run[key]
    worker["exit_code"] = process.poll()


def completed(worker, state):
    path = Path(worker["prefix"] + "_run.json")
    run = json.loads(path.read_text())
    if worker.get("exit_code") != 0 or run.get("status") != "passed":
        raise ValueError(f"GPU {worker['gpu']} did not complete")
    if (run["rounds"] != state["rounds"] or run["external_mode"] != state["external_mode"]
            or run["gpu"] != worker["gpu"] or run["gpu_uuid"] != worker["gpu_uuid"]
            or run["completed_shapes"] != worker["shape_count"]
            or {tuple(s) for s in run["shapes"]} != {tuple(s) for s in worker["shapes"]}):
        raise ValueError("Worker identity or shape coverage mismatch")
    if (run["protected_sha256"].get(str(HERE / "benchmark.py")) != state["benchmark_sha256"]
            or run["protected_sha256"].get(state["experiments_file"]) != state["experiments_sha256"]):
        raise ValueError("Worker used a different harness or experiment configuration")
    for kind, digest in run["output_sha256"].items():
        if bench.base.sha256(run["outputs"][kind]) != digest:
            raise ValueError(f"Worker output changed: GPU {worker['gpu']} {kind}")
    worker.update(run_manifest=str(path), run_sha256=bench.base.sha256(path))
    return run


def stop(processes):
    for process in processes:
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
    deadline = time.monotonic() + 15
    while any(p.poll() is None for p in processes) and time.monotonic() < deadline:
        time.sleep(0.2)
    for process in processes:
        if process.poll() is None:
            process.terminate()
    for process in processes:
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", type=Path, default=HERE / "shapes295.csv")
    parser.add_argument("--experiments", type=Path, default=HERE / "experiments.json")
    parser.add_argument("--batch", required=True, help="fresh child directory of this harness")
    parser.add_argument("--rounds", type=int, choices=(3, 5), default=3)
    parser.add_argument("--external-mode", choices=("full", "finalists"), default="full")
    parser.add_argument("--full-batch", type=Path)
    parser.add_argument("--gpus", default=",".join(map(str, sorted(bench.GPU_MAP))))
    parser.add_argument("--plan-only", action="store_true", help="print the shard plan; do not start measurements")
    args = parser.parse_args()
    if not args.batch or Path(args.batch).name != args.batch or args.batch in {".", ".."}:
        parser.error("--batch must be a simple fresh directory name")
    if (args.external_mode == "finalists") != (args.full_batch is not None):
        parser.error("--full-batch is required exactly in finalists mode")
    gpus = [int(value) for value in args.gpus.split(",")]
    if not gpus or len(gpus) != len(set(gpus)) or not set(gpus) <= bench.GPU_MAP.keys():
        parser.error("--gpus must contain unique mapped physical GPU indices")
    args.shapes, args.experiments = args.shapes.resolve(), args.experiments.resolve()
    if args.full_batch:
        args.full_batch = args.full_batch.resolve()
    shapes = bench.read_shapes(args.shapes)
    batch_dir = HERE / args.batch
    workers = plan(shapes, gpus, args.external_mode, args.full_batch, batch_dir)
    state = dict(status="prepared", pid=os.getpid(), shapes=shapes, shape_count=len(shapes),
                 shapes_file=str(args.shapes), shapes_sha256=bench.base.sha256(args.shapes),
                 rounds=args.rounds, external_mode=args.external_mode, workers=workers,
                 experiments_file=str(args.experiments), experiments_sha256=bench.base.sha256(args.experiments),
                 full_batch=str(args.full_batch) if args.full_batch else None,
                 benchmark_sha256=bench.base.sha256(HERE / "benchmark.py"),
                 launcher_sha256=bench.base.sha256(__file__), jit_dir=str(bench.JIT))
    if args.plan_only:
        print(json.dumps(state, indent=2))
        return 0
    from build import verify_build
    verify_build()
    # Read-only prerequisites: no compilation and no separate validation jobs.
    state["libraries"] = bench.experiments(args.experiments)
    required = [bench.JIT / f"{name}.so" for name in bench.base.MODULES]
    if any(not path.is_file() for path in required):
        raise FileNotFoundError("Existing frozen JIT modules are required")
    state["preflight"] = preflight.check(gpus, bench.GPU_MAP)
    batch_dir.mkdir(exist_ok=False)
    for worker in workers:
        with Path(worker["shapes_file"]).open("x", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("M", "N", "K"))
            writer.writerows(worker["shapes"])
    manifest_path = batch_dir / "launch.json"
    state.update(status="running", start_time=time.time())
    bench.atomic_json(manifest_path, state)
    processes = []
    def save():
        state["updated_time"] = time.time()
        bench.atomic_json(manifest_path, state)
    def interrupted(_signum, _frame):
        raise KeyboardInterrupt("Launcher received a termination signal")
    signal.signal(signal.SIGTERM, interrupted)
    try:
        for worker in workers:
            env = os.environ.copy()
            for key in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
                env.pop(key, None)
            env["OPUS_TUNE_GPU_INDEX"] = str(worker["gpu"])
            command = [sys.executable, "-u", str(HERE / "benchmark.py"), "--shapes", worker["shapes_file"],
                       "--prefix", worker["prefix"], "--rounds", str(args.rounds),
                       "--external-mode", args.external_mode, "--experiments", str(args.experiments)]
            if worker["full_manifest"]:
                command += ["--full-manifest", worker["full_manifest"]]
            with Path(worker["prefix"] + ".log").open("x") as stream:
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
            processes.append(process)
            worker.update(pid=process.pid, command=command, status="starting")
            save()
        while True:
            for worker, process in zip(workers, processes):
                refresh(worker, process)
            save()
            print(json.dumps({"workers": [{key: worker.get(key) for key in
                  ("gpu", "status", "completed_shapes", "timing_records", "rejected_checks", "exit_code")}
                  for worker in workers]}), flush=True)
            if all(p.poll() is not None for p in processes):
                break
            time.sleep(10)
        runs, errors = [], []
        for worker in workers:
            try:
                runs.append(completed(worker, state))
            except Exception as exc:
                worker.update(status="failed", error=repr(exc))
                errors.append(repr(exc))
        state.update(status="failed" if errors else "passed", errors=errors,
                     completed_shapes=sum(r["completed_shapes"] for r in runs),
                     opus_wins=sum(r.get("opus_wins", 0) for r in runs))
    except BaseException as exc:
        state.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                     error=repr(exc), traceback=traceback.format_exc())
        stop(processes)
    finally:
        state["end_time"] = time.time()
        save()
    print(json.dumps({"status": state["status"], "completed_shapes": state.get("completed_shapes", 0),
                      "manifest": str(manifest_path)}), flush=True)
    return 0 if state["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
