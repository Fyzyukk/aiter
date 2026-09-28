#!/usr/bin/env python3
"""Shard target shapes across available physical GPUs and remeasure finalist IDs."""

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

import resume_benchmark as resume
import launch as original

bench = resume.bench
HERE, HARNESS, ROOT = resume.HERE, resume.HARNESS, resume.bench.ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", type=Path, default=HERE / "shapes99.csv")
    parser.add_argument("--experiments", type=Path, default=HERE / "experiments_v1.json")
    parser.add_argument("--full-batch", type=Path, default=HARNESS / "full295_r3")
    parser.add_argument("--batch", default="resume99_v1_r3", help="fresh child directory of this generalization experiment")
    parser.add_argument("--rounds", type=int, choices=(3, 5), default=3)
    parser.add_argument("--gpus", default="4,5,6,7", help="physical GPU indices")
    parser.add_argument("--plan-only", action="store_true", help="validate artifacts and print plan; no GPU work")
    args = parser.parse_args()
    if not args.batch or Path(args.batch).name != args.batch or args.batch in {".", ".."}:
        parser.error("--batch must be a simple fresh directory name")
    gpus = [int(value) for value in args.gpus.split(",")]
    if not gpus or len(gpus) != len(set(gpus)) or not set(gpus) <= bench.GPU_MAP.keys():
        parser.error("--gpus must contain unique mapped physical GPU indices")
    args.shapes, args.experiments = args.shapes.resolve(), args.experiments.resolve()
    args.full_batch = args.full_batch.resolve()
    shapes = bench.read_shapes(args.shapes)
    batch_dir = HERE / args.batch
    if batch_dir.exists():
        raise FileExistsError(f"Use a fresh batch: {batch_dir}")
    workers = original.plan(shapes, gpus, "full", None, batch_dir)
    _, _, references = bench.base.snapshot(bench.JIT)
    references["measurement_harness"] = bench.base.sha256(HARNESS / "benchmark.py")
    references["archived_harness_helpers"] = bench.base.sha256(bench.ARCHIVE / "benchmark.py")
    source_launch = args.full_batch / "launch.json"
    _, _, transfer, source_hashes = resume.transferred_finalists(source_launch, references, shapes, gpus[0])
    for worker in workers:
        worker["full_manifest"] = str(source_launch)
    state = dict(status="prepared", pid=os.getpid(), shapes=shapes, shape_count=len(shapes),
                 shapes_file=str(args.shapes), shapes_sha256=bench.base.sha256(args.shapes),
                 rounds=args.rounds, external_mode="finalists", workers=workers,
                 experiments_file=str(args.experiments), experiments_sha256=bench.base.sha256(args.experiments),
                 full_batch=str(args.full_batch), benchmark_sha256=bench.base.sha256(HARNESS / "benchmark.py"),
                 launcher_sha256=bench.base.sha256(__file__), jit_dir=str(bench.JIT),
                 finalist_transfer_policy=transfer["policy"],
                 finalist_source_sha256=source_hashes,
                 resume_benchmark_sha256=bench.base.sha256(HERE / "resume_benchmark.py"),
                 original_launcher_sha256=bench.base.sha256(HARNESS / "launch.py"))
    state["libraries"] = bench.experiments(args.experiments)
    if args.plan_only:
        print(json.dumps(state, indent=2))
        return 0
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
        raise KeyboardInterrupt("Resume launcher received a termination signal")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        for worker in workers:
            env = os.environ.copy()
            for key in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
                env.pop(key, None)
            env["OPUS_TUNE_GPU_INDEX"] = str(worker["gpu"])
            command = [sys.executable, "-u", str(HERE / "resume_benchmark.py"),
                       "--shapes", worker["shapes_file"], "--prefix", worker["prefix"],
                       "--rounds", str(args.rounds), "--external-mode", "finalists",
                       "--experiments", str(args.experiments), "--full-manifest", str(source_launch)]
            with Path(worker["prefix"] + ".log").open("x") as stream:
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
            processes.append(process)
            worker.update(pid=process.pid, command=command, status="starting")
            save()
        while True:
            for worker, process in zip(workers, processes):
                original.refresh(worker, process)
            save()
            print(json.dumps({"workers": [{key: worker.get(key) for key in
                  ("gpu", "status", "completed_shapes", "timing_records", "rejected_checks", "exit_code")}
                  for worker in workers]}), flush=True)
            if all(process.poll() is not None for process in processes):
                break
            time.sleep(10)
        runs, errors = [], []
        for worker in workers:
            try:
                run = original.completed(worker, state)
                if (run["protected_sha256"].get(str(HERE / "resume_benchmark.py")) != state["resume_benchmark_sha256"]
                        or run["protected_sha256"].get(str(HERE / "resume_launch.py")) != state["launcher_sha256"]
                        or run["finalist_transfer"]["actual_gpu"] != worker["gpu"]):
                    raise ValueError("Worker used a different resume wrapper or GPU")
                runs.append(run)
            except Exception as exc:
                worker.update(status="failed", error=repr(exc))
                errors.append(repr(exc))
        state.update(status="failed" if errors else "passed", errors=errors,
                     completed_shapes=sum(run["completed_shapes"] for run in runs),
                     opus_wins=sum(run.get("opus_wins", 0) for run in runs))
    except BaseException as exc:
        state.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                     error=repr(exc), traceback=traceback.format_exc())
        original.stop(processes)
    finally:
        state["end_time"] = time.time()
        save()
    print(json.dumps({"status": state["status"], "completed_shapes": state.get("completed_shapes", 0),
                      "manifest": str(manifest_path)}), flush=True)
    return 0 if state["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
