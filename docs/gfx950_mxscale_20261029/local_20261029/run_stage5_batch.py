#!/usr/bin/env python3
"""Run a fresh OPUS plan on PCI-verified physical GPU5/6, then audit six rounds.

Two-round screening is only candidate selection. Its shortlist always measures
each selected candidate and the step4 control again in a fresh six-round batch.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone

LOCAL = Path(__file__).resolve().parent
MAPPING = {2: (6, "0000:e5:00.0"), 3: (5, "0000:95:00.0")}


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def csv_write(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(plan, output, rounds, iters):
    from recover_final_gpu56 import preflight
    output.mkdir()
    (output / "repeats").mkdir()
    (output / "worker_plans").mkdir()
    preflight(output)
    dump(output / "repeat_plan.json", plan)
    subprocess.run([sys.executable, str(LOCAL / "precompile_stage5_plan.py"),
                    "--plan", str(output / "repeat_plan.json"),
                    "--receipt", str(output / "compile_receipt.json")], check=True)
    csv_write(output / "expected.csv", [item["shape"] for item in plan["shapes"]])
    known = []
    for item in plan["shapes"]:
        for candidate in item["candidates"]:
            # Experimental candidates may legitimately fail correctness. Every
            # planned round must still be recorded; only the known control is
            # required to remain correct for a completed experiment audit.
            if "current_step4_best_control" not in candidate.get("roles", []):
                continue
            known.append({**item["shape"], **{field: candidate[field] for field in
                          ("kid", "kernelName", "kind", "traits")},
                          "geometry": json.dumps(candidate["geometry"])})
    csv_write(output / "known.csv", known)
    shards, costs = {2: [], 3: []}, {2: 0., 3: 0.}
    def cost(item):
        shape = item["shape"]
        return len(item["candidates"]) * max(1., shape["M"] * shape["N"] * shape["K"] / 1e10)
    for item in sorted(plan["shapes"], key=cost, reverse=True):
        gpu = min(costs, key=costs.get)
        shards[gpu].append(item)
        costs[gpu] += cost(item)
    processes, streams = {}, []
    try:
        for gpu, items in shards.items():
            if not items:
                continue
            physical, pci = MAPPING[gpu]
            worker_plan = output / f"worker_plans/gpu{gpu}_plan.json"
            dump(worker_plan, {**plan, "shapes": items, "gpu_physical_id": physical,
                               "gpu_pci_bus": pci})
            command = [sys.executable, str(LOCAL / "repeat_opus_candidates.py"),
                       "--plan", str(worker_plan), "--output", str(output / f"repeats/gpu{gpu}.jsonl"),
                       "--gpu", str(gpu), "--rounds", str(rounds), "--iters", str(iters)]
            subprocess.run(command + ["--dry-run"], check=True)
            log = (output / f"repeats/gpu{gpu}.log").open("x")
            streams.append(log)
            processes[gpu] = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            print(f"Started logical GPU{gpu}/physical GPU{physical}: {len(items)} shapes, pid {processes[gpu].pid}", flush=True)
        statuses = {gpu: process.wait() for gpu, process in processes.items()}
    except BaseException:
        for process in processes.values():
            if process.poll() is None:
                process.terminate()
        raise
    finally:
        for stream in streams:
            stream.close()
    dump(output / "worker_status.json", statuses)
    if any(statuses.values()):
        raise RuntimeError(f"Workers failed: {statuses}")
    if rounds == 6:
        subprocess.run([sys.executable, str(LOCAL / "audit_final_repeat.py"),
                        "--plan", str(output / "repeat_plan.json"),
                        "--files", str(output / "repeats/gpu*.jsonl"),
                        "--expected", str(output / "expected.csv"),
                        "--known-csv", str(output / "known.csv"),
                        "--output-dir", str(output / "audit"),
                        "--physical-gpus", "7,4,6,5", "--require-complete"], check=True)
    return output


def shortlist(plan, screen):
    records = defaultdict(list)
    for path in (screen / "repeats").glob("gpu*.jsonl"):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            key = tuple(row["shape"][field] for field in ("gfx", "cu_num", "M", "N", "K", "w_scale_block"))
            records[(key, row["kid"])].append(row)
    selected = []
    for item in plan["shapes"]:
        shape = tuple(item["shape"][field] for field in ("gfx", "cu_num", "M", "N", "K", "w_scale_block"))
        def timing(candidate):
            rows = records[(shape, candidate["kid"])]
            if len(rows) != 2 or any(not row["correct"] or row["failure"] for row in rows):
                return float("inf")
            return statistics.median(row["us"] for row in rows)
        ranked = sorted(item["candidates"], key=timing)
        keep = {candidate["kid"] for candidate in ranked[:5] if timing(candidate) != float("inf")}
        families = set()
        for candidate in ranked:
            family = candidate["geometry"][0]
            if family not in families and timing(candidate) != float("inf"):
                keep.add(candidate["kid"])
                families.add(family)
            if "current_step4_best_control" in candidate.get("roles", []):
                keep.add(candidate["kid"])
        # Preserve a matched body/schedule counterpart only when one side
        # survived selection, so isolated kernel changes remain attributable.
        available = {c["kid"] for c in ranked if timing(c) != float("inf")}
        for pair in plan.get("matched_controls", []):
            if keep.intersection(pair):
                keep.update(set(pair).intersection(available))
        selected.append({**item, "candidates": [candidate for candidate in item["candidates"] if candidate["kid"] in keep]})
    return {**plan, "selection": "Top5 and per-pipeline best from two-round selection plus exact step4 control; fresh six rounds",
            "screen_source": str(screen), "shapes": selected}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--screen-first", action="store_true")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    args.output.mkdir()
    dump(args.output / "controller.json", {"pid": os.getpid(), "started_utc": datetime.now(timezone.utc).isoformat(),
                                          "plan": str(args.plan), "mode": "screen_then_six" if args.screen_first else "six"})
    if args.screen_first:
        screen = run(plan, args.output / "screen", 2, 21)
        plan = shortlist(plan, screen)
    run(plan, args.output / "confirm", 6, 101)
    dump(args.output / "completed.json", {"completed_utc": datetime.now(timezone.utc).isoformat(), "exit_code": 0})


if __name__ == "__main__":
    main()
