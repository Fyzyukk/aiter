#!/usr/bin/env python3
"""CPU-only six-round CO/JIT repeat accounting and historical CO restoration.

Reports initial full-profile and repeated-shortlist winners independently.
The historical CO cohort has three separate repeated CSV-win counts: best OPUS,
best CO, and the exact original CO configuration matched by its complete name.
"""

import argparse
import glob
import json
import statistics
from collections import defaultdict
from pathlib import Path

from summarize_step1_repeat import (KEYS, boolean, csv_rows, jsonl_rows, key,
                                    number, shape_row, valid_result, write_csv)

LOCAL = Path(__file__).resolve().parent
DATA = LOCAL.parents[2] / "docs/gfx950_mxscale_20261029/handoff_data"


def candidate_statistics(shape, kid, candidate, records, baseline, args):
    expected_rounds = set(range(args.rounds))
    complete = set(records) == expected_rounds
    correct_rows = [row for row in records.values()
                    if boolean(row.get("correct")) and valid_result(row, args.err_ratio)]
    times = [float(row["us"]) for row in correct_rows]
    correct_all = complete and len(correct_rows) == args.rounds
    timing_matches = complete and all(int(row.get("rotate_args", 0)) == 0 for row in records.values())
    median = statistics.median(times) if times else None
    return {**shape_row(shape), "kid": kid, "kernelName": candidate["kernelName"],
            "kind": candidate["kind"], "roles": json.dumps(candidate.get("roles", [])),
            "rounds_written": len(records), "rounds_complete": complete,
            "correct_rounds": len(correct_rows), "all_rounds_correct": correct_all,
            "timing_matches_tuner": timing_matches, "median_us": median,
            "min_us": min(times) if times else None, "max_us": max(times) if times else None,
            "pstdev_us": statistics.pstdev(times) if times else None,
            "max_errRatio": max((number(row.get("errRatio")) or 0 for row in records.values()), default=None),
            "csv_baseline_us": baseline,
            "wins_csv_median": bool(correct_all and timing_matches and baseline is not None and median < baseline),
            "wins_csv_every_round": bool(correct_all and timing_matches and baseline is not None and max(times) < baseline),
            "median_csv_speedup": baseline / median if correct_all and timing_matches and baseline is not None else None,
            "initial_status": candidate.get("status"), "initial_us": candidate.get("us"),
            "initial_correct": candidate.get("correct"),
            "gpu_logical_id": next((row.get("gpu_logical_id") for row in records.values() if "gpu_logical_id" in row), None)}


def best(rows, *, kind=None, kids=None):
    available = [row for row in rows if row["all_rounds_correct"] and row["timing_matches_tuner"]
                 and (kind is None or row["kind"] == kind) and (kids is None or row["kid"] in kids)]
    return min(available, key=lambda row: row["median_us"]) if available else None


def best_fields(prefix, result):
    return {prefix + "_kid": result["kid"] if result else None,
            prefix + "_name": result["kernelName"] if result else None,
            prefix + "_kind": result["kind"] if result else None,
            prefix + "_median_us": result["median_us"] if result else None,
            prefix + "_max_us": result["max_us"] if result else None,
            prefix + "_wins_csv_median": bool(result and result["wins_csv_median"]),
            prefix + "_wins_csv_every_round": bool(result and result["wins_csv_every_round"])}


def analyze(args):
    expected = {key(row) for row in csv_rows(args.expected)}
    baseline = {key(row): float(row["us"]) for row in csv_rows(args.baseline)}
    plan = json.loads(args.plan.read_text())
    if int(plan.get("version", 0)) != 2:
        raise ValueError("Use a generic version2 prepare_opus_repeat plan")
    plan_items = {key(item["shape"]): item for item in plan["shapes"]}
    if set(plan_items) != expected:
        raise ValueError(f"Plan shape coverage differs: missing={len(expected-set(plan_items))}, extra={len(set(plan_items)-expected)}")
    planned = {shape: {int(candidate["kid"]): candidate for candidate in item["candidates"]}
               for shape, item in plan_items.items()}
    paths = []
    for pattern in args.files or [str(LOCAL / "step2_repeat_shard*.jsonl")]:
        matches = glob.glob(pattern)
        if not matches:
            raise ValueError(f"No repeat JSONL matched {pattern}")
        paths.extend(Path(path).resolve() for path in matches)
    paths = sorted(set(paths))
    rounds_expected = set(range(args.rounds))
    records = defaultdict(dict)
    shape_files, shape_gpus = defaultdict(set), defaultdict(set)
    for path in paths:
        for event, location in jsonl_rows(path):
            shape, kid, round_id = key(event["shape"]), int(event["kid"]), int(event["round"])
            if shape not in planned or kid not in planned[shape]:
                raise ValueError(f"Unplanned shape/kid at {location}")
            candidate = planned[shape][kid]
            if event["kernelName"] != candidate["kernelName"] or event.get("kind", candidate["kind"]) != candidate["kind"]:
                raise ValueError(f"Candidate identity differs from plan at {location}")
            if "roles" in event and event["roles"] != candidate.get("roles", []):
                raise ValueError(f"Candidate roles differ from plan at {location}")
            if round_id not in rounds_expected:
                raise ValueError(f"Unexpected round {round_id} at {location}")
            if round_id in records[(shape, kid)]:
                raise ValueError(f"Duplicate shape/kid/round at {location}")
            records[(shape, kid)][round_id] = event
            shape_files[shape].add(str(path))
            if "gpu_logical_id" in event:
                shape_gpus[shape].add(int(event["gpu_logical_id"]))
    if any(len(shape_files[shape]) > 1 or len(shape_gpus[shape]) > 1 for shape in expected):
        raise ValueError("A shape was split between files or GPUs")
    # Current names and kind are taken from the pure registry, never historical IDs.
    from prepare_opus_repeat import registry_metadata
    sample_shape = next(iter(expected))
    metadata, _ = registry_metadata(family=plan["family"], gfx=sample_shape[0], cu_num=sample_shape[1])
    initial_profile = defaultdict(dict)
    for row in csv_rows(args.profile):
        if row.get("libtype") != "opus" or key(row) not in expected:
            continue
        shape, kid = key(row), int(row["kernelId"])
        if kid not in metadata:
            raise ValueError(f"Initial profile contains unavailable current kid {kid}")
        if number(row.get("us")) is not None and float(row["us"]) > 0 and row.get("kernelName") != metadata[kid]["kernelName"]:
            raise ValueError(f"Initial profile identity differs for current kid {kid}")
        initial_profile[shape][kid] = row
    initial_progress = {}
    for event, _ in jsonl_rows(args.progress):
        shape = key(event["shape"])
        if shape in expected:
            initial_progress[shape] = event
    candidates, shapes, old_details = [], [], []
    for shape in sorted(expected):
        csv_us = baseline.get(shape)
        current = [candidate_statistics(shape, kid, candidate, records[(shape, kid)], csv_us, args)
                   for kid, candidate in planned[shape].items()]
        candidates.extend(current)
        remote = plan_items[shape]["remote_winner"]
        mapping = remote.get("mapping", {})
        matched_kids = set(int(kid) for kid in mapping.get("legal_current_kids", remote.get("mapped_current_kids", [])))
        best_opus, best_co, best_jit = best(current), best(current, kind="CO"), best(current, kind="JIT")
        best_old = best(current, kids=matched_kids)
        valid_initial = [(kid, row) for kid, row in initial_profile[shape].items() if valid_result(row, args.err_ratio)]
        initial_best = min(valid_initial, key=lambda item: float(item[1]["us"])) if valid_initial else None
        initial_co = [(kid, row) for kid, row in valid_initial if metadata[kid]["kind"] == "CO"]
        initial_best_co = min(initial_co, key=lambda item: float(item[1]["us"])) if initial_co else None
        initial_old = [(kid, row) for kid, row in valid_initial if kid in matched_kids]
        initial_best_old = min(initial_old, key=lambda item: float(item[1]["us"])) if initial_old else None
        complete = bool(current) and all(row["rounds_complete"] for row in current)
        row = {**shape_row(shape), "csv_baseline_us": csv_us,
               "remote_winner_name": remote["kernelName"], "remote_reference_us": remote["us"],
               "historical_winner_kind": "CO" if remote["is_co"] else "JIT",
               "remote_mapping_method": mapping.get("method"), "remote_mapped": bool(matched_kids),
               "remote_mapped_current_kids": json.dumps(sorted(matched_kids)),
               "initial_tune_complete": shape in initial_progress,
               "initial_profile_candidates": len(initial_profile[shape]),
               "initial_best_kid": initial_best[0] if initial_best else None,
               "initial_best_us": float(initial_best[1]["us"]) if initial_best else None,
               "initial_best_wins_csv": bool(initial_best and csv_us is not None and float(initial_best[1]["us"]) < csv_us),
               "initial_best_co_kid": initial_best_co[0] if initial_best_co else None,
               "initial_best_co_us": float(initial_best_co[1]["us"]) if initial_best_co else None,
               "initial_best_co_wins_csv": bool(initial_best_co and csv_us is not None and float(initial_best_co[1]["us"]) < csv_us),
               "initial_exact_old_kid": initial_best_old[0] if initial_best_old else None,
               "initial_exact_old_us": float(initial_best_old[1]["us"]) if initial_best_old else None,
               "initial_exact_old_wins_csv": bool(initial_best_old and csv_us is not None and float(initial_best_old[1]["us"]) < csv_us),
               "repeat_complete": complete, "planned_candidates": len(current),
               "correct_all_rounds_candidates": sum(item["all_rounds_correct"] for item in current),
               "exact_old_any_correct_all_rounds": any(item["kid"] in matched_kids and item["all_rounds_correct"] for item in current),
               "any_opus_wins_csv_every_round": any(item["wins_csv_every_round"] for item in current),
               "any_co_wins_csv_every_round": any(item["kind"] == "CO" and item["wins_csv_every_round"] for item in current),
               "any_exact_old_wins_csv_every_round": any(item["kid"] in matched_kids and item["wins_csv_every_round"] for item in current)}
        for prefix, winner in (("best_repeat", best_opus), ("best_repeat_co", best_co),
                               ("best_repeat_jit", best_jit), ("exact_old_repeat", best_old)):
            row.update(best_fields(prefix, winner))
        row["new_best_vs_exact_old_pct"] = ((best_old["median_us"] / best_opus["median_us"] - 1) * 100
                                            if best_old and best_opus else None)
        row["repeat_status"] = "pending" if not complete else "no_correct_candidate" if not best_opus else "wins_csv" if best_opus["wins_csv_median"] else "slower_than_csv"
        shapes.append(row)
        for kid in sorted(matched_kids):
            candidate = next((item for item in current if item["kid"] == kid), None)
            initial_row = initial_profile[shape].get(kid)
            old_details.append({**shape_row(shape), "remote_winner_name": remote["kernelName"],
                                "remote_reference_us": remote["us"], "historical_winner_kind": row["historical_winner_kind"],
                                "mapping_method": mapping.get("method"), "current_kid": kid,
                                "current_name": metadata[kid]["kernelName"], "current_kind": metadata[kid]["kind"],
                                "initial_correct": bool(initial_row and valid_result(initial_row, args.err_ratio)),
                                "initial_us": number(initial_row.get("us")) if initial_row else None,
                                "initial_errRatio": number(initial_row.get("errRatio")) if initial_row else None,
                                "rounds_complete": bool(candidate and candidate["rounds_complete"]),
                                "correct_rounds": candidate["correct_rounds"] if candidate else 0,
                                "median_us": candidate["median_us"] if candidate else None,
                                "min_us": candidate["min_us"] if candidate else None,
                                "max_us": candidate["max_us"] if candidate else None,
                                "csv_baseline_us": csv_us,
                                "wins_csv_median": bool(candidate and candidate["wins_csv_median"]),
                                "wins_csv_every_round": bool(candidate and candidate["wins_csv_every_round"])})
    cohorts = {}
    for kind in ("CO", "JIT"):
        group = [row for row in shapes if row["historical_winner_kind"] == kind]
        cohorts[kind] = {"historical_winner_shapes": len(group), "mapped_complete_winner_shapes": sum(row["remote_mapped"] for row in group),
                         "repeat_complete_shapes": sum(row["repeat_complete"] for row in group),
                         "repeat_exact_old_winner_all_rounds_correct_shapes": sum(row["exact_old_any_correct_all_rounds"] for row in group),
                         "initial_best_opus_wins_csv": sum(row["initial_best_wins_csv"] for row in group),
                         "initial_best_co_wins_csv": sum(row["initial_best_co_wins_csv"] for row in group),
                         "initial_exact_old_winner_wins_csv": sum(row["initial_exact_old_wins_csv"] for row in group),
                         "repeat_best_opus_wins_csv_median": sum(row["best_repeat_wins_csv_median"] for row in group),
                         "repeat_best_co_wins_csv_median": sum(row["best_repeat_co_wins_csv_median"] for row in group),
                         "repeat_exact_old_winner_wins_csv_median": sum(row["exact_old_repeat_wins_csv_median"] for row in group),
                         "repeat_any_opus_wins_csv_every_round": sum(row["any_opus_wins_csv_every_round"] for row in group),
                         "repeat_any_co_wins_csv_every_round": sum(row["any_co_wins_csv_every_round"] for row in group),
                         "repeat_exact_old_winner_wins_csv_every_round": sum(row["any_exact_old_wins_csv_every_round"] for row in group)}
    losses = [row for row in shapes if not row["best_repeat_wins_csv_median"]]
    summary = {"expected_shapes": len(expected), "initial_completed_shapes": len(initial_progress),
               "initial_best_opus_wins_csv": sum(row["initial_best_wins_csv"] for row in shapes),
               "repeat_complete_shapes": sum(row["repeat_complete"] for row in shapes),
               "repeat_correct_candidate_shapes": sum(row["correct_all_rounds_candidates"] > 0 for row in shapes),
               "repeat_exact_old_winner_all_rounds_correct_shapes": sum(row["exact_old_any_correct_all_rounds"] for row in shapes),
               "repeat_wins_csv_median": sum(row["best_repeat_wins_csv_median"] for row in shapes),
               "repeat_wins_csv_every_round": sum(row["any_opus_wins_csv_every_round"] for row in shapes),
               "remaining_losses": len(losses), "rounds": args.rounds, "correctness_limit": args.err_ratio,
               "planned_candidates": len(candidates),
               "candidate_groups_complete": sum(row["rounds_complete"] for row in candidates),
               "candidate_groups_correct_all_rounds": sum(row["all_rounds_correct"] for row in candidates),
               "historical_winner_cohorts": cohorts, "files": [str(path) for path in paths],
               "definitions": {"median_win": "Candidate is correct in every expected round and median_us < fixed CSV baseline",
                               "every_round_win": "Candidate is correct in every expected round and max_us < fixed CSV baseline",
                               "historical_source": "plan.remote_winner is the same-round all_shapes_421.csv opus_kernel/opus_us; old_kernel and historical kid are unused",
                               "best_co_scope": "Initial best CO searches the full profile; repeated best CO searches the shortlist, which explicitly includes initial best_co"}}
    return {"summary": summary, "shapes": shapes, "candidates": candidates,
            "remote_winners": old_details, "remaining_losses": losses,
            "historical_co_shapes": [row for row in shapes if row["historical_winner_kind"] == "CO"],
            "historical_jit_shapes": [row for row in shapes if row["historical_winner_kind"] == "JIT"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", help="Repeat shard JSONL paths or quoted globs")
    parser.add_argument("--plan", type=Path, default=LOCAL / "step2_paired/repeat_plan.json")
    parser.add_argument("--profile", type=Path, default=LOCAL / "step2_profile.csv")
    parser.add_argument("--progress", type=Path, default=LOCAL / "step2_progress.jsonl")
    parser.add_argument("--expected", type=Path, default=DATA / "regressions_46_shapes.csv")
    parser.add_argument("--baseline", type=Path, default=DATA / "csv_baseline.csv")
    parser.add_argument("--output-dir", type=Path, default=LOCAL / "step2_report")
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--err-ratio", type=float, default=0.05)
    args = parser.parse_args()
    if args.rounds < 1 or not 0 <= args.err_ratio <= 1:
        parser.error("rounds must be positive and err-ratio in [0,1]")
    try:
        report = analyze(args)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for name in ("shapes", "candidates", "remote_winners", "remaining_losses", "historical_co_shapes", "historical_jit_shapes"):
        write_csv(args.output_dir / f"{name}.csv", report[name])
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
