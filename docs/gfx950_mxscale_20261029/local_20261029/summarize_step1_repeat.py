#!/usr/bin/env python3
"""CPU-only accounting for stage 1 tuning and disjoint six-round GPU shards.

Example:
  python3 summarize_step1_repeat.py 'step1_repeat_shard*.jsonl'

Every-round wins are reported separately from median wins. Exact guard pairs
are compared within the same shape and round. Candidate identity is its current
name and kid, and the four shards must not duplicate any shape/kid/round.
"""

import argparse
import csv
import glob
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

LOCAL = Path(__file__).resolve().parent
ROOT = LOCAL.parents[2]
DATA = ROOT / "docs/gfx950_mxscale_20261029/handoff_data"
KEYS = ("gfx", "cu_num", "M", "N", "K", "w_scale_block")


def csv_rows(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def key(row):
    return (str(row.get("gfx", "gfx950")), int(row.get("cu_num", 256)),
            int(row["M"]), int(row["N"]), int(row["K"]),
            str(row.get("w_scale_block", "128x128")))


def shape_row(shape):
    return dict(zip(KEYS, shape))


def number(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def boolean(value):
    return value is True or str(value).lower() == "true"


def valid_result(row, err_ratio):
    latency, error = number(row.get("us")), number(row.get("errRatio"))
    return latency is not None and latency > 0 and error is not None and 0 <= error <= err_ratio


def jsonl_rows(path):
    with Path(path).open() as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line), f"{path}:{line_number}"
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid or unfinished JSONL at {path}:{line_number}: {exc}") from exc


def write_csv(path, rows):
    columns = list(dict.fromkeys(column for row in rows for column in row))
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def analyze(args):
    expected = {key(row) for row in csv_rows(args.expected)}
    baseline = {key(row): float(row["us"]) for row in csv_rows(args.baseline)}
    plan = json.loads(args.plan.read_text())
    planned = {key(item["shape"]): {int(candidate["kid"]): candidate for candidate in item["candidates"]}
               for item in plan["shapes"]}
    if set(planned) != expected:
        raise ValueError(f"Plan coverage differs: missing={len(expected-set(planned))}, extra={len(set(planned)-expected)}")
    paths = []
    for pattern in args.files or [str(LOCAL / "step1_repeat_shard*.jsonl")]:
        matches = sorted(glob.glob(pattern))
        if not matches:
            raise ValueError(f"No repeat JSONL matched {pattern}")
        paths.extend(Path(path).resolve() for path in matches)
    paths = sorted(set(paths))
    initial = {}
    for event, _ in jsonl_rows(args.progress):
        shape = key(event["shape"])
        if shape in expected:
            initial[shape] = event
    rounds_expected = set(range(args.rounds))
    observations = defaultdict(dict)
    source_files = defaultdict(set)
    gpu_ids = defaultdict(set)
    for path in paths:
        for event, location in jsonl_rows(path):
            shape, kid, round_id = key(event["shape"]), int(event["kid"]), int(event["round"])
            if shape not in planned or kid not in planned[shape]:
                raise ValueError(f"Unexpected shape/kid at {location}")
            if round_id not in rounds_expected:
                raise ValueError(f"Unexpected round {round_id} at {location}")
            if event["kernelName"] != planned[shape][kid]["kernelName"]:
                raise ValueError(f"Candidate name differs from plan at {location}")
            group = observations[(shape, kid)]
            if round_id in group:
                raise ValueError(f"Duplicate shape/kid/round across shards at {location}")
            group[round_id] = event
            source_files[shape].add(str(path))
            if "gpu_logical_id" in event:
                gpu_ids[shape].add(int(event["gpu_logical_id"]))
    for shape in expected:
        if len(source_files[shape]) > 1 or len(gpu_ids[shape]) > 1:
            raise ValueError(f"Shape was shared across multiple shards or GPUs: {shape}")
    candidates = []
    candidate_by_shape = defaultdict(list)
    for shape in sorted(expected):
        for kid, candidate in planned[shape].items():
            group = observations[(shape, kid)]
            complete = set(group) == rounds_expected
            good = [row for row in group.values() if boolean(row.get("correct")) and valid_result(row, args.err_ratio)]
            latencies = [float(row["us"]) for row in good]
            correct_all = complete and len(good) == args.rounds
            mode_matches = complete and all(int(row.get("rotate_args", 0)) == 0 for row in group.values())
            csv_us = baseline.get(shape)
            median = statistics.median(latencies) if latencies else None
            median_win = correct_all and mode_matches and csv_us is not None and median < csv_us
            every_win = correct_all and mode_matches and csv_us is not None and max(latencies) < csv_us
            row = {**shape_row(shape), "kid": kid, "kernelName": candidate["kernelName"],
                   "bounded": candidate["bounded"], "rounds_written": len(group),
                   "complete": complete, "correct_rounds": len(good), "all_rounds_correct": correct_all,
                   "timing_matches_tuner": mode_matches, "median_us": median,
                   "min_us": min(latencies) if latencies else None, "max_us": max(latencies) if latencies else None,
                   "pstdev_us": statistics.pstdev(latencies) if latencies else None,
                   "csv_baseline_us": csv_us, "wins_csv_median": median_win, "wins_csv_every_round": every_win,
                   "median_csv_speedup": csv_us / median if correct_all and mode_matches and csv_us is not None else None,
                   "gpu_logical_id": next(iter(gpu_ids[shape])) if gpu_ids[shape] else None}
            candidates.append(row)
            candidate_by_shape[shape].append(row)
    pair_initial = {(key(row), int(row["false_kid"])): row for row in csv_rows(args.paired_dir / "pairs_by_shape.csv")}
    targets = {int(row["false_kid"]): int(row["true_kid"]) for row in csv_rows(args.paired_dir / "pair_metadata.csv")}
    target_pairs = []
    for (shape, false_kid), old_row in sorted(pair_initial.items()):
        if shape not in expected or false_kid not in targets:
            continue
        true_kid = targets[false_kid]
        # Same tile geometry alone does not imply shape legality: panel64 has a
        # smaller K ceiling than panel128. The plan includes measured legal pairs.
        if false_kid not in planned[shape] and true_kid not in planned[shape]:
            continue
        false_group, true_group = observations[(shape, false_kid)], observations[(shape, true_kid)]
        false_good = {r: row for r, row in false_group.items() if boolean(row.get("correct")) and valid_result(row, args.err_ratio)}
        true_good = {r: row for r, row in true_group.items() if boolean(row.get("correct")) and valid_result(row, args.err_ratio)}
        common = sorted(false_good.keys() & true_good.keys())
        ratios = [float(true_good[r]["us"]) / float(false_good[r]["us"]) for r in common]
        paired_complete = set(false_group) == rounds_expected and set(true_group) == rounds_expected
        both_correct = paired_complete and len(ratios) == args.rounds
        ratio = statistics.median(ratios) if ratios else None
        row = {**shape_row(shape), "false_kid": false_kid, "true_kid": true_kid,
               "initial_false_status": old_row.get("false_status"), "initial_true_status": old_row.get("true_status"),
               "initial_false_us": number(old_row.get("false_us")), "initial_true_us": number(old_row.get("true_us")),
               "paired_rounds_complete": paired_complete, "false_correct_rounds": len(false_good),
               "true_correct_rounds": len(true_good), "both_correct_rounds": len(ratios), "all_rounds_both_correct": both_correct,
               "median_true_over_false": ratio, "min_true_over_false": min(ratios) if ratios else None,
               "max_true_over_false": max(ratios) if ratios else None,
               "median_false_advantage_pct": (ratio - 1) * 100 if ratio is not None else None,
               "min_false_advantage_pct": (min(ratios) - 1) * 100 if ratios else None,
               "max_false_advantage_pct": (max(ratios) - 1) * 100 if ratios else None,
               "false_faster_rounds": sum(value > 1 for value in ratios),
               "false_faster_every_round": both_correct and min(ratios) > 1,
               "false_margin_every_round": both_correct and min(ratios) >= 1 + args.margin_pct / 100,
               "csv_baseline_us": baseline.get(shape),
               "false_median_us": statistics.median(float(row["us"]) for row in false_good.values()) if false_good else None,
               "true_median_us": statistics.median(float(row["us"]) for row in true_good.values()) if true_good else None}
        target_pairs.append(row)
    false_configs = []
    for false_kid, true_kid in sorted(targets.items()):
        rows = [row for row in target_pairs if row["false_kid"] == false_kid]
        medians = [row["median_false_advantage_pct"] for row in rows if row["all_rounds_both_correct"]]
        false_configs.append({"false_kid": false_kid, "true_kid": true_kid, "applicable_shapes": len(rows),
                              "complete_pairs": sum(row["paired_rounds_complete"] for row in rows),
                              "false_correct_all_rounds_shapes": sum(row["false_correct_rounds"] == args.rounds for row in rows),
                              "true_correct_all_rounds_shapes": sum(row["true_correct_rounds"] == args.rounds for row in rows),
                              "both_correct_all_rounds_shapes": sum(row["all_rounds_both_correct"] for row in rows),
                              "false_faster_every_round_shapes": sum(row["false_faster_every_round"] for row in rows),
                              "false_margin_every_round_shapes": sum(row["false_margin_every_round"] for row in rows),
                              "median_shape_advantage_pct": statistics.median(medians) if medians else None,
                              "min_shape_advantage_pct": min(medians) if medians else None,
                              "max_shape_advantage_pct": max(medians) if medians else None})
    shapes = []
    for shape in sorted(expected):
        rows = candidate_by_shape[shape]
        usable = [row for row in rows if row["all_rounds_correct"] and row["timing_matches_tuner"]]
        winner = min(usable, key=lambda row: row["median_us"]) if usable else None
        complete = bool(rows) and all(row["complete"] for row in rows)
        event = initial.get(shape)
        opus = event.get("best_opus") if event else None
        initial_correct = bool(opus and valid_result(opus, args.err_ratio))
        csv_us = baseline.get(shape)
        initial_win = initial_correct and csv_us is not None and float(opus["us"]) < csv_us
        median_win = bool(winner and winner["wins_csv_median"])
        every_win = any(row["wins_csv_every_round"] for row in usable)
        row = {**shape_row(shape), "csv_baseline_us": csv_us, "initial_tune_completed": event is not None,
               "initial_opus_correct": initial_correct, "initial_opus_us": float(opus["us"]) if initial_correct else None,
               "initial_wins_csv": initial_win, "repeat_complete": complete,
               "planned_candidates": len(rows), "correct_all_rounds_candidates": len(usable),
               "best_repeat_kid": winner["kid"] if winner else None,
               "best_repeat_name": winner["kernelName"] if winner else None,
               "best_repeat_median_us": winner["median_us"] if winner else None,
               "best_repeat_max_us": winner["max_us"] if winner else None,
               "best_repeat_csv_speedup": winner["median_csv_speedup"] if winner else None,
               "repeat_wins_csv_median": median_win, "repeat_wins_csv_every_round": every_win,
               "reason": "pending_repeat" if not complete else "no_correct_repeat_candidate" if not winner
               else "missing_csv_baseline" if csv_us is None else "wins_csv" if median_win else "slower_than_csv"}
        shapes.append(row)
    losses = [row for row in shapes if not row["repeat_wins_csv_median"]]
    initial_losses = [row for row in shapes if not row["initial_wins_csv"]]
    summary = {"expected_shapes": len(expected), "initial_completed_shapes": len(initial),
               "initial_correct_opus_shapes": sum(row["initial_opus_correct"] for row in shapes),
               "initial_wins_csv": sum(row["initial_wins_csv"] for row in shapes),
               "repeat_completed_shapes": sum(row["repeat_complete"] for row in shapes),
               "repeat_correct_candidate_shapes": sum(row["correct_all_rounds_candidates"] > 0 for row in shapes),
               "repeat_wins_csv_median": sum(row["repeat_wins_csv_median"] for row in shapes),
               "repeat_wins_csv_every_round": sum(row["repeat_wins_csv_every_round"] for row in shapes),
               "repeat_remaining_losses": len(losses), "rounds": args.rounds,
               "planned_candidates": len(candidates), "candidate_groups_with_all_rounds": sum(row["complete"] for row in candidates),
               "candidate_groups_correct_all_rounds": sum(row["all_rounds_correct"] for row in candidates),
               "target_false_configurations": len(targets), "target_shape_pairs_expected": len(target_pairs),
               "target_shape_pairs_complete": sum(row["paired_rounds_complete"] for row in target_pairs),
               "target_shape_pairs_both_correct": sum(row["all_rounds_both_correct"] for row in target_pairs),
               "target_false_faster_every_round_pairs": sum(row["false_faster_every_round"] for row in target_pairs),
               "target_false_margin_every_round_pairs": sum(row["false_margin_every_round"] for row in target_pairs),
               "stable_margin_pct": args.margin_pct,
               "remaining_losses_by_n_k": dict(Counter(f'N{row["N"]}/K{row["K"]}' for row in losses).most_common()),
               "files": [str(path) for path in paths], "correctness_limit": args.err_ratio,
               "definitions": {"median_win": "One candidate correct in every round has median_us < fixed CSV us",
                               "every_round_win": "One candidate correct in every round has max_us < fixed CSV us",
                               "false_faster_every_round": "Both configurations correct in every round and each paired true_us/false_us > 1",
                               "false_margin_every_round": "Each paired true_us/false_us >= 1 + stable_margin_pct / 100",
                               "scope": "Repeat shortlist contains m-repeat candidates; initial OPUS winners come from full targeted tune"}}
    return {"summary": summary, "shapes": shapes, "candidates": candidates, "target_pairs": target_pairs,
            "false_configurations": false_configs, "remaining_losses": losses, "initial_losses": initial_losses}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", help="Four repeat JSONL paths or quoted glob patterns")
    parser.add_argument("--progress", type=Path, default=LOCAL / "step1_progress.jsonl")
    parser.add_argument("--expected", type=Path, default=DATA / "m_repeat_32_shapes.csv")
    parser.add_argument("--baseline", type=Path, default=DATA / "csv_baseline.csv")
    parser.add_argument("--paired-dir", type=Path, default=LOCAL / "step1_paired")
    parser.add_argument("--plan", type=Path, default=LOCAL / "step1_paired/repeat_plan.json")
    parser.add_argument("--output-dir", type=Path, default=LOCAL / "step1_report")
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--err-ratio", type=float, default=0.05)
    parser.add_argument("--margin-pct", type=float, default=1.0)
    args = parser.parse_args()
    if args.rounds < 1 or not 0 <= args.err_ratio <= 1 or args.margin_pct < 0:
        parser.error("rounds must be positive, err-ratio in [0,1], margin-pct nonnegative")
    try:
        report = analyze(args)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for name in ("shapes", "candidates", "target_pairs", "false_configurations", "remaining_losses", "initial_losses"):
        write_csv(args.output_dir / f"{name}.csv", report[name])
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
