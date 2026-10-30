#!/usr/bin/env python3
"""CPU-only stage3 alternated medians, pipeline ordering and parent details."""

import argparse
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path

from prepare_opus_repeat import (LOCAL, DATA, csv_rows, key, registry_metadata,
                                 pure_common, shape_row, write_csv)
from summarize_step1_repeat import jsonl_rows
from summarize_opus_repeat import candidate_statistics, best, best_fields
from analyze_stage3_existing import metadata_details, strict_peers


def analyze(args):
    plan = json.loads(args.plan.read_text())
    expected = {key(item["shape"]): item for item in plan["shapes"]}
    if args.expected and set(expected) != {key(row) for row in csv_rows(args.expected)}:
        raise ValueError("Repeat plan differs from expected shape file")
    if int(plan.get("version", 0)) != 2:
        raise ValueError("Version2 current-registry repeat plan required")
    metadata, _ = registry_metadata()
    common = pure_common()
    registry = common.get_registry("a8w8_blockscale_bpreshuffle", "gfx950", 256)
    peers = strict_peers(registry)
    baseline = {key(row): float(row["us"]) for row in csv_rows(args.baseline)}
    planned = {shape: {int(candidate["kid"]): candidate for candidate in item["candidates"]}
               for shape, item in expected.items()}
    files = set()
    for pattern in args.files:
        matches = glob.glob(pattern)
        if not matches:
            raise ValueError(f"No files match {pattern}")
        files.update(Path(path).resolve() for path in matches)
    rounds, shape_files, shape_gpus = defaultdict(dict), defaultdict(set), defaultdict(set)
    for path in sorted(files):
        for event, location in jsonl_rows(path):
            shape, kid, index = key(event["shape"]), int(event["kid"]), int(event["round"])
            if shape not in planned or kid not in planned[shape]:
                raise ValueError(f"Unplanned shape/kid at {location}")
            candidate = planned[shape][kid]
            if kid not in metadata or candidate["kernelName"] != metadata[kid]["kernelName"] or candidate["traits"] != metadata[kid]["traits"]:
                raise ValueError(f"Plan differs from current registry at {location}")
            if event["kernelName"] != candidate["kernelName"] or event.get("roles", []) != candidate.get("roles", []):
                raise ValueError(f"Repeat identity/roles differ from plan at {location}")
            if index not in range(args.rounds) or index in rounds[(shape, kid)]:
                raise ValueError(f"Duplicate or unexpected round at {location}")
            rounds[(shape, kid)][index] = event
            shape_files[shape].add(str(path))
            shape_gpus[shape].add(event.get("gpu_logical_id"))
    if any(len(shape_files[s]) > 1 or len(shape_gpus[s]) > 1 for s in expected):
        raise ValueError("A shape was split across GPUs or files")
    candidate_rows, shape_rows, all_4wave_rows = [], [], []
    for shape, item in sorted(expected.items()):
        current = []
        for kid, candidate in planned[shape].items():
            stat = candidate_statistics(shape, kid, candidate, rounds[(shape, kid)], baseline[shape], args)
            details = metadata_details(registry[kid], metadata[kid])
            stat.update({k: v for k, v in details.items() if k not in stat})
            stat["strict_base_trait_peers"] = peers.get(kid, [])
            current.append(stat)
        candidate_rows.extend(current)
        valid = sorted((r for r in current if r["all_rounds_correct"] and r["timing_matches_tuner"]),
                       key=lambda r: (r["median_us"], r["kid"]))
        best_opus = best(current)
        by_pipeline = defaultdict(list)
        by_kind = defaultdict(list)
        for row in valid:
            by_pipeline[row["pipeline"]].append(row)
            by_kind[(row["pipeline"], row["kind"])].append(row)
        fourwave = by_pipeline.get("pin_agpr", [])
        all_4wave_rows.extend(fourwave)
        shape_info = {**shape_row(shape), "csv_baseline_us": baseline[shape],
                      "repeat_complete": bool(current) and all(r["rounds_complete"] for r in current),
                      "planned_candidates": len(current),
                      "correct_all_rounds_candidates": sum(r["all_rounds_correct"] for r in current),
                      "repeat_status": "pending", **best_fields("best_repeat", best_opus),
                      "best_by_pipeline": {pipeline: values[:3] for pipeline, values in by_pipeline.items()},
                      "best_by_pipeline_kind": {pipeline + ":" + kind: values[:3]
                                                for (pipeline, kind), values in by_kind.items()},
                      "all_4wave_parents": fourwave,
                      "known_4wave_planned": sum(r["pipeline"] == "pin_agpr" for r in current),
                      "known_4wave_correct": len(fourwave)}
        if shape_info["repeat_complete"]:
            shape_info["repeat_status"] = "no_correct_candidate" if best_opus is None else "wins_csv" if best_opus["wins_csv_median"] else "slower_than_csv"
        shape_rows.append(shape_info)
    complete = [row for row in shape_rows if row["repeat_complete"]]
    losses = [row for row in complete if row["repeat_status"] != "wins_csv"]
    summary = {"expected_shapes": len(expected), "repeat_complete_shapes": len(complete),
               "repeat_wins_csv_median": sum(row["repeat_status"] == "wins_csv" for row in complete),
               "remaining_losses": len(losses), "pending_shapes": len(expected) - len(complete),
               "rounds": args.rounds, "planned_candidates": len(candidate_rows),
               "candidate_groups_complete": sum(row["rounds_complete"] for row in candidate_rows),
               "candidate_groups_correct_all_rounds": sum(row["all_rounds_correct"] for row in candidate_rows),
               "pipeline_winners": dict(Counter(registry[row["best_repeat_kid"]].bpreshuffle_co["pipeline"]
                                                 if registry[row["best_repeat_kid"]].bpreshuffle_co else
                                                 registry[row["best_repeat_kid"]].bpreshuffle_entry["pipeline"]
                                                 for row in complete if row["best_repeat_kid"] is not None)),
               "loss_NK": dict(Counter(f"N{row['N']}/K{row['K']}" for row in losses)),
               "files": [str(path) for path in sorted(files)],
               "definitions": {"median_win": "Every round correct, cache rotation matches tuner, median OPUS us < fixed CSV baseline",
                               "parent_scope": "All legal known 4wave parents and shortlisted B/overlap CO/JIT, paired by complete base traits"}}
    return {"summary": summary, "shapes": shape_rows, "candidates": candidate_rows,
            "all_4wave_parents": all_4wave_rows, "remaining_losses": losses}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", help="Repeat shard paths or quoted glob")
    parser.add_argument("--plan", type=Path, default=LOCAL / "step3_existing_round1/cpu_analysis/repeat_plan.json")
    parser.add_argument("--expected", type=Path)
    parser.add_argument("--baseline", type=Path, default=DATA / "csv_baseline.csv")
    parser.add_argument("--output-dir", type=Path, default=LOCAL / "step3_existing_round1/repeat_report")
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--err-ratio", type=float, default=.05)
    args = parser.parse_args()
    report = analyze(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (args.output_dir / "summary.json").write_text(json.dumps(report["summary"], indent=2, allow_nan=False) + "\n")
    for name in ("shapes", "candidates", "all_4wave_parents", "remaining_losses"):
        rows = [{k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v
                 for k, v in row.items()} for row in report[name]]
        if rows:
            write_csv(args.output_dir / (name + ".csv"), rows)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
