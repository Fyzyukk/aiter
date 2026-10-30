#!/usr/bin/env python3
"""Validate and run a complete shortlist on the four authorized GPUs.

Source env.sh before invoking this script. Each shape stays on one GPU and
uses the existing balanced six-round runner. No existing output is replaced.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

LOCAL = Path(__file__).resolve().parent
FIELDS = ("gfx", "cu_num", "M", "N", "K", "w_scale_block")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--physical-gpus", default="0,1,2,3", help="Four authorized physical GPUs; logical indices remain 0-3")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.rounds < 2 or args.rounds % 2:
        parser.error("Use a positive even number of rounds")
    try:
        physical_gpus = [int(value) for value in args.physical_gpus.split(",")]
    except ValueError:
        parser.error("--physical-gpus must contain four comma-separated GPU IDs")
    if len(physical_gpus) != 4 or len(set(physical_gpus)) != 4 or any(gpu not in range(8) for gpu in physical_gpus):
        parser.error("Select four distinct physical GPUs from 0 to 7")
    if not args.dry_run and (os.environ.get("HIP_VISIBLE_DEVICES") != "0,1,2,3"
                            or os.environ.get("ROCR_VISIBLE_DEVICES") != ",".join(map(str, physical_gpus))):
        parser.error("HIP_VISIBLE_DEVICES must be logical 0,1,2,3; ROCR_VISIBLE_DEVICES must match --physical-gpus")
    plan = json.loads(args.plan.read_text())
    items = plan["shapes"]
    keys = [tuple(item["shape"][field] for field in FIELDS) for item in items]
    if not items or len(keys) != len(set(keys)) or any(not item["candidates"] for item in items):
        parser.error("Plan must contain distinct shapes and nonempty candidate lists")
    if args.out_dir.exists():
        parser.error("Choose a fresh output directory to preserve completed rounds")
    args.out_dir.mkdir(parents=True)
    shards, loads = [[], [], [], []], [0.0] * 4
    def cost(item):
        # Include correctness/reference work as well as timed kernels.
        shape = item["shape"]
        return len(item["candidates"]) * max(1, shape["M"] * shape["N"] * shape["K"] / 1e10)
    for item in sorted(items, key=cost, reverse=True):
        gpu = min(range(4), key=lambda index: (loads[index], len(shards[index])))
        shards[gpu].append(item)
        loads[gpu] += cost(item)
    runner = LOCAL / "repeat_opus_candidates.py"
    commands = []
    for gpu, shard in enumerate(shards):
        if not shard:
            continue
        shard_path = args.out_dir / f"gpu{gpu}_plan.json"
        shard_path.write_text(json.dumps({**plan, "shapes": shard,
                                         "gpu_logical_id": gpu,
                                         "gpu_physical_id": physical_gpus[gpu]}, indent=2) + "\n")
        command = [sys.executable, str(runner), "--plan", str(shard_path),
                   "--output", str(args.out_dir / f"gpu{gpu}.jsonl"),
                   "--gpu", str(gpu), "--rounds", str(args.rounds)]
        subprocess.run(command + ["--dry-run"], check=True)
        commands.append((gpu, command))
    receipt = {"shapes": len(items), "candidates": sum(len(item["candidates"]) for item in items),
               "rounds": args.rounds, "gpu_ids": [gpu for gpu, _ in commands],
               "physical_gpu_ids": physical_gpus,
               "source_plan": str(args.plan.resolve()), "phase": "validated"}
    (args.out_dir / "status.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)
    if args.dry_run:
        return
    processes, streams = [], []
    try:
        for gpu, command in commands:
            stream = (args.out_dir / f"gpu{gpu}.log").open("x")
            streams.append(stream)
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
            processes.append((gpu, process))
            (args.out_dir / f"gpu{gpu}.pid").write_text(f"{process.pid}\n")
        statuses = {gpu: process.wait() for gpu, process in processes}
    except BaseException:
        for _, process in processes:
            if process.poll() is None:
                process.terminate()
        for _, process in processes:
            process.wait()
        raise
    finally:
        for stream in streams:
            stream.close()
    receipt.update(phase="complete" if all(code == 0 for code in statuses.values()) else "failed",
                   return_codes=statuses)
    (args.out_dir / "status.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)
    if any(statuses.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
