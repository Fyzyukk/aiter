#!/usr/bin/env python3
"""Concise, CPU-only status for the current shape-grouped tuning run."""
import argparse
import csv
import json
from pathlib import Path

LOCAL = Path(__file__).resolve().parent
ROOT = LOCAL.parents[2]
FIELDS = ("gfx", "cu_num", "M", "N", "K", "w_scale_block")


def shape_key(row):
    return tuple(str(row[field]) for field in FIELDS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--expected", type=int, required=True)
    parser.add_argument("--jit", action="store_true")
    args = parser.parse_args()
    with (LOCAL.parent / "handoff_data/csv_baseline.csv").open() as stream:
        baseline = {shape_key(row): float(row["us"]) for row in csv.DictReader(stream)}
    rows = {}
    progress = args.run_dir / "progress.jsonl"
    if progress.exists():
        for line in progress.read_text().splitlines():
            try:
                row = json.loads(line)
                rows[shape_key(row["shape"])] = row
            except (ValueError, KeyError):
                pass
    wins = sum(bool(row.get("best_opus")) and row["best_opus"]["us"] < baseline[key]
               for key, row in rows.items())
    invalid = sum(not row.get("best_opus") for row in rows.values())
    active = []
    for path in sorted((args.run_dir / "candidates").glob("*.json")):
        try:
            row = json.loads(path.read_text())
        except ValueError:
            continue
        keys = tuple(map(str, row.get("info_keys", ())))
        if keys in rows:
            continue
        active.append({"gpu": row["gpu_id"], "shape": row["info_keys"][2:5],
                       "processed": row["processed_total"], "total": row["total_tasks"]})
    result = {"completed": len(rows), "expected": args.expected, "wins_csv": wins,
              "losses_csv": len(rows) - wins - invalid, "no_correct_opus": invalid,
              "active": active}
    if args.jit:
        build = ROOT / "aiter/jit/build/module_deepgemm_opus/build"
        ninja = build / "build.ninja"
        if ninja.exists():
            targets = [line.split(":", 1)[0][6:] for line in ninja.read_text().splitlines()
                       if line.startswith("build ") and ": cuda_compile " in line]
            result["jit_objects"] = [sum((build / target).exists() for target in targets), len(targets)]
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
