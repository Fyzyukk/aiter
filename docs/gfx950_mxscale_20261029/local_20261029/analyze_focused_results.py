#!/usr/bin/env python3
"""CPU-only focused screen/six-round accounting against the fixed CSV.

Screen measurements never become six-round claims. Repeat results use only
the supplied current batch, all planned candidates, explicit rotate_args=0,
and the frozen focused selection receipt for the previous18/new38 cohorts.
Historical records are references and never fill missing current rounds.
"""

import argparse
import glob
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import finish_workflow as accounting
import prepare_opus_repeat as metadata_helper
from prepare_co_experiment import ROOT, LOCAL, load_common

FIELDS = accounting.SHAPE_FIELDS
CONFIRMED = (100003, 100024, 100025, 100030, 100031)
PARENTS = {100003: 389, 100024: 100001, 100025: 100003, 100030: 100022, 100031: 100023}


def dump(path, value):
    accounting.atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def usable(row):
    return bool(row and row.get("correct") and row.get("us") is not None and row["us"] > 0)


def group_summary(rows):
    counts = Counter(row["status"] for row in rows)
    return {"expected": len(rows), "completed": len(rows) - counts["pending"],
            "wins_csv": counts["wins_csv"], "losses_csv": counts["slower_than_csv"],
            "no_correct_opus": counts["no_correct_opus"], "pending": counts["pending"]}


def repeat_results(args, expected, baseline, metadata, common):
    plan = json.loads(args.plan.read_text())
    current_registry = common.get_registry("a8w8_blockscale_bpreshuffle", "gfx950", 256)
    if plan.get("version") != 2:
        raise ValueError("Current version2 repeat plan required")
    planned, locations = {}, {}
    for item in plan["shapes"]:
        shape = accounting.strict_key(item["shape"], args.plan)
        if shape in planned:
            raise ValueError("Duplicate planned shape")
        planned[shape] = {}
        for row in item["candidates"]:
            kid = int(row["kid"])
            if kid in planned[shape] or kid not in metadata:
                raise ValueError("Duplicate/unavailable planned candidate")
            for field in ("kernelName", "kind", "traits", "geometry"):
                if row.get(field) != metadata[kid][field]:
                    raise ValueError(f"Plan/current {field} identity differs for kid{kid}")
            if not common.a8w8_mxscale_bpreshuffle_supports_shape(current_registry[kid], *shape[2:5]):
                raise ValueError("Planned current candidate is shape-illegal")
            planned[shape][kid] = row
    if set(planned) != expected:
        raise ValueError("This repeat plan must exactly cover the expected focused subset")
    files = sorted({Path(path).resolve() for pattern in (args.files or [str(args.run_dir / "repeats/gpu*.jsonl")])
                    for path in glob.glob(str(pattern))})
    if not files:
        raise ValueError("No current repeat JSONL files matched")
    records, shape_files, shape_gpus, positions = defaultdict(dict), defaultdict(set), defaultdict(set), defaultdict(set)
    partial = []
    for path in files:
        for event, location in accounting.jsonl(path, partial):
            shape = accounting.strict_key(event["shape"], location)
            kid, rnd = event.get("kid"), event.get("round")
            if shape not in planned or type(kid) is not int or kid not in planned[shape] or type(rnd) is not int or rnd not in range(6):
                raise ValueError(f"Unplanned shape/candidate/round at {location}")
            candidate = planned[shape][kid]
            if event.get("kernelName") != candidate["kernelName"] or event.get("kind") != candidate["kind"] or event.get("roles", []) != candidate.get("roles", []):
                raise ValueError(f"Current repeat identity/roles differs at {location}")
            if event.get("csv_baseline_us") != baseline[shape]:
                raise ValueError("Saved repeat baseline differs from the fixed six-field CSV")
            if type(event.get("rotate_args")) is not int or event["rotate_args"] != 0:
                raise ValueError("Current timing must explicitly prove rotate_args=0")
            gpu, position = event.get("gpu_logical_id"), event.get("position")
            if type(gpu) is not int or gpu not in range(4) or type(position) is not int or position not in range(len(planned[shape])):
                raise ValueError("Missing/invalid GPU0-3 or current candidate position")
            if rnd in records[(shape, kid)] or position in positions[(shape, rnd)]:
                raise ValueError("Duplicate current candidate round/position")
            records[(shape, kid)][rnd] = event
            positions[(shape, rnd)].add(position)
            shape_files[shape].add(str(path))
            shape_gpus[shape].add(gpu)
    if any(len(shape_files[shape]) > 1 or len(shape_gpus[shape]) > 1 for shape in expected):
        raise ValueError("A current shape spans different files/GPUs; do not mix batches")
    candidates, shape_rows = {}, []
    for shape in sorted(expected):
        current = {}
        for kid, planned_row in planned[shape].items():
            rounds = records[(shape, kid)]
            complete = set(rounds) == set(range(6))
            correct = complete and all(accounting.valid(row, .05) and type(row.get("correct")) is bool
                                       and not row.get("failure") for row in rounds.values())
            times = [float(row["us"]) for row in rounds.values()] if correct else []
            current[kid] = {"kid": kid, "kernelName": metadata[kid]["kernelName"],
                "pipeline": metadata[kid]["geometry"][0], "correct": correct,
                "us": statistics.median(times) if times else None, "min_us": min(times) if times else None,
                "max_us": max(times) if times else None,
                "errRatio": max(float(row["errRatio"]) for row in rounds.values()) if correct else None,
                "rounds_complete": complete, "rounds_seen": sorted(rounds), "evidence_kind": "current_six_round_median"}
        candidates[shape] = current
        complete = bool(current) and all(row["rounds_complete"] for row in current.values())
        if complete and any(positions[(shape, rnd)] != set(range(len(current))) for rnd in range(6)):
            raise ValueError("A completed current round has missing candidate positions")
        valid = [row for row in current.values() if usable(row)]
        best = min(valid, key=lambda row: (row["us"], row["kid"])) if valid else None
        shape_rows.append({**accounting.shape_row(shape), "status": "pending" if not complete else
            "no_correct_opus" if not best else "wins_csv" if best["us"] < baseline[shape] else "slower_than_csv",
            "kid": best["kid"] if best else None, "kernelName": best["kernelName"] if best else None,
            "us": best["us"] if best else None, "errRatio": best["errRatio"] if best else None,
            "csv_baseline_us": baseline[shape], "wins_csv": bool(complete and best and best["us"] < baseline[shape]),
            "wins_csv_every_round": bool(complete and best and best["max_us"] < baseline[shape]),
            "repeat_complete": complete, "planned_candidates": len(current),
            "correct_candidates": len(valid), "evidence_kind": "current_six_round_median" if complete and best else None})
    return shape_rows, candidates, records, {"files": list(map(str, files)), "partial_final_lines": partial,
        "planned_candidate_shape_groups": sum(len(items) for items in planned.values()),
        "candidate_groups_complete": sum(row["rounds_complete"] for items in candidates.values() for row in items.values()),
        "candidate_groups_correct_all_six": sum(row["correct"] for items in candidates.values() for row in items.values()),
        "plan": str(args.plan)}, planned


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("screen", "repeat"), required=True)
    parser.add_argument("--run-dir", type=Path, default=LOCAL / "focused_expansion_known_complete")
    parser.add_argument("--expected", type=Path)
    parser.add_argument("--selection-receipt", type=Path)
    parser.add_argument("--known-winners", type=Path)
    parser.add_argument("--baseline", type=Path, default=metadata_helper.DATA / "csv_baseline.csv")
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--progress", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--files", nargs="*", help="Current repeat JSONLs or quoted globs only")
    parser.add_argument("--registry-repo", type=Path, default=ROOT, help="Public repo; archived clone only for explicit CPU validation")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    args.expected = args.expected or args.run_dir / "shapes.csv"
    args.selection_receipt = args.selection_receipt or args.run_dir / "receipt.json"
    args.known_winners = args.known_winners or args.run_dir / "known_winners.csv"
    args.profile = args.profile or args.run_dir / "profile.csv"
    args.progress = args.progress or args.run_dir / "progress.jsonl"
    args.plan = args.plan or args.run_dir / "final_repeat_plan.json"
    args.output_dir = args.output_dir or LOCAL / "focused_result_analysis" / args.phase
    expected_rows = metadata_helper.csv_rows(args.expected)
    expected = {accounting.strict_key(row, args.expected) for row in expected_rows}
    if len(expected) != len(expected_rows):
        raise ValueError("Duplicate focused six-field shape")
    selection = json.loads(args.selection_receipt.read_text())
    previous = {accounting.strict_key(row, args.selection_receipt): row for row in selection["shapes"]}
    if not expected <= previous.keys():
        raise ValueError("Focused expected shapes are not covered by the frozen selection receipt")
    baseline = {}
    for row in metadata_helper.csv_rows(args.baseline):
        shape = accounting.strict_key(row, args.baseline, require_scope=False)
        if shape not in expected:
            continue
        if shape in baseline or accounting.finite(row["us"]) is None or float(row["us"]) <= 0:
            raise ValueError("Duplicate/invalid fixed CSV baseline")
        baseline[shape] = float(row["us"])
        if previous[shape]["csv_baseline_us"] != baseline[shape]:
            raise ValueError("Frozen selection and fixed CSV baseline disagree")
    if set(baseline) != expected:
        raise ValueError("Missing fixed six-field baseline")
    common = load_common(args.registry_repo.resolve(), label="focused_results")
    metadata_helper.pure_common = lambda: common
    metadata, _ = metadata_helper.registry_metadata()
    known = defaultdict(set)
    for row in metadata_helper.csv_rows(args.known_winners):
        shape = accounting.strict_key({"gfx": "gfx950", "cu_num": 256, "w_scale_block": "128x128", **row}, args.known_winners)
        if shape not in expected:
            continue
        kid = int(row["kid"])
        if kid not in metadata or row["kernelName"] != metadata[kid]["kernelName"] or json.loads(row["geometry"]) != metadata[kid]["geometry"] or row["traits"] != metadata[kid]["traits"]:
            raise ValueError("Frozen known winner has unavailable/changed full identity")
        known[shape].add(kid)
    if args.phase == "screen":
        namespace = SimpleNamespace(expected=args.expected, baseline=args.baseline, profile=[str(args.profile)],
            progress=[str(args.progress)] if args.progress.exists() else [], full_progress=[], fallback_progress=[],
            fallback_expected=None, repeat=[], rounds=6, err_ratio=.05, current_registry=True, filter_to_expected=True)
        report, final = accounting.analyze(namespace)
        shapes, candidates, raw, planned = report["shapes"], {}, {}, None
        for shape, items in final.items():
            candidates[shape] = {row["kernelId"]: {**row, "kid": row["kernelId"], "evidence_kind": "screen_single_pass"}
                                 for row in items.values()}
        for row in shapes:
            row["kid"] = row["kernelId"]
            row["evidence_kind"] = "screen_single_pass" if row["us"] is not None else None
        batch = report["summary"]
    else:
        shapes, candidates, raw, batch, planned = repeat_results(args, expected, baseline, metadata, common)
    comparisons, directions, known_missing = [], [], []
    for row in shapes:
        shape = accounting.strict_key(row, "result")
        old = previous[shape]
        row.update(selection_class=old["selection_class"],
                   previous_complete_gpu_measurement=old["previous_complete_gpu_measurement"],
                   prior_us=old["best_current_us"], prior_kid=old["best_current_kid"], prior_wins_csv=old["current_wins_csv"],
                   csv_gap_pct=(row["us"] / baseline[shape] - 1) * 100 if row["us"] else None)
        comparison = {**row, "gain_vs_prior_pct": (old["best_current_us"] / row["us"] - 1) * 100
                      if old["best_current_us"] and row["us"] else None}
        comparison["win_transition"] = ("pending" if row["status"] == "pending" else
            "new_current_validation" if not old["previous_complete_gpu_measurement"] else
            "kept_win" if old["current_wins_csv"] and row["wins_csv"] else
            "lost_win" if old["current_wins_csv"] else "recovered_win" if row["wins_csv"] else "still_loss")
        comparisons.append(comparison)
        current = candidates.get(shape, {})
        for kid in known[shape]:
            reason = "not_measured" if kid not in current else "incorrect_or_incomplete" if not usable(current[kid]) else None
            if reason:
                known_missing.append({**accounting.shape_row(shape), "kid": kid, "reason": reason})
        for kid in CONFIRMED:
            if kid not in current:
                continue
            new, parent_id = current[kid], PARENTS[kid]
            parent = current.get(parent_id)
            controls = [current[other] for other in known[shape] - set(CONFIRMED) if other in current and usable(current[other])]
            control = min(controls, key=lambda candidate: candidate["us"]) if controls else None
            info = {**accounting.shape_row(shape), "new_kid": kid, "exact_parent_kid": parent_id,
                "phase": args.phase, "new_us": new["us"], "correct": new["correct"],
                "csv_baseline_us": baseline[shape], "wins_csv": bool(usable(new) and new["us"] < baseline[shape]),
                "parent_status": "unmeasured" if parent is None else "correct" if usable(parent) else "incorrect_or_incomplete",
                "parent_us": parent["us"] if parent else None,
                "gain_vs_parent_pct": (parent["us"] / new["us"] - 1) * 100 if usable(parent) and usable(new) else None,
                "best_known_control_kid": control["kid"] if control else None,
                "best_known_control_us": control["us"] if control else None,
                "gain_vs_known_control_pct": (control["us"] / new["us"] - 1) * 100 if control and usable(new) else None}
            if args.phase == "repeat" and usable(parent) and usable(new):
                ratios = [raw[(shape, parent_id)][rnd]["us"] / raw[(shape, kid)][rnd]["us"] for rnd in range(6)]
                order = [raw[(shape, kid)][rnd]["position"] < raw[(shape, parent_id)][rnd]["position"] for rnd in range(6)]
                if any(order[rnd] == order[rnd + 1] for rnd in (0, 2, 4)):
                    raise ValueError("Exact new/parent round-pair ordering was not reversed")
                info.update(paired_median_gain_pct=(statistics.median(ratios) - 1) * 100,
                            paired_round_gain_pct=[(ratio - 1) * 100 for ratio in ratios],
                            faster_than_parent_rounds=sum(ratio > 1 for ratio in ratios))
            directions.append(info)
    losses = [row for row in comparisons if row["status"] in ("slower_than_csv", "no_correct_opus")]
    nk, mc = defaultdict(list), defaultdict(list)
    for row in comparisons:
        nk[f"N{row['N']}/K{row['K']}"].append(row)
        mc[accounting.m_class(row["M"])].append(row)
    old39 = [row for row in comparisons if row["selection_class"] in ("initial_observed_loss", "historical_pending_loss")]
    prior18 = [row for row in comparisons if row["previous_complete_gpu_measurement"]]
    extra38 = [row for row in comparisons if not row["previous_complete_gpu_measurement"]]
    summary = {"phase": args.phase, **group_summary(comparisons),
        "initial39": group_summary(old39), "initial_observed": group_summary([row for row in old39 if row["selection_class"] == "initial_observed_loss"]),
        "historical_pending": group_summary([row for row in old39 if row["selection_class"] == "historical_pending_loss"]),
        "initial39_recovery": {"previously_current_wins": sum(row["prior_wins_csv"] for row in old39),
            "new_win_after_prior_current_loss": sum(row["previous_complete_gpu_measurement"] and not row["prior_wins_csv"] and row["wins_csv"] for row in old39),
            "first_current_win_from_historical_pending": sum(not row["previous_complete_gpu_measurement"] and row["wins_csv"] for row in old39),
            "prior_current_win_retained": sum(row["prior_wins_csv"] and row["wins_csv"] for row in old39)},
        "previous18": group_summary(prior18), "new38": group_summary(extra38),
        "prior18_transitions": dict(Counter(row["win_transition"] for row in prior18)),
        "by_NK": {group: group_summary(rows) for group, rows in sorted(nk.items())},
        "by_M_class": {group: group_summary(rows) for group, rows in mc.items()},
        "remaining_M_by_NK": {group: sorted(row["M"] for row in rows if row["status"] in ("slower_than_csv", "no_correct_opus")) for group, rows in sorted(nk.items())},
        "known_candidate_failures_or_pending": len(known_missing), "batch": batch,
        "by_confirmed_direction": {str(kid): {"measured_shapes": len(items),
            "correct_shapes": sum(row["correct"] for row in items), "candidate_wins_csv": sum(row["wins_csv"] for row in items),
            "better_than_exact_parent": sum(row["gain_vs_parent_pct"] is not None and row["gain_vs_parent_pct"] > 0 for row in items),
            "missing_or_incorrect_parent": sum(row["parent_status"] != "correct" for row in items),
            "faster_all_six_parent_rounds": sum(row.get("faster_than_parent_rounds") == 6 for row in items)}
            for kid in CONFIRMED for items in [[row for row in directions if row["new_kid"] == kid]]},
        "screen_only": args.phase == "screen", "gpu_used_by_tool": False,
        "definitions": {"screen": "Single21-iteration initial screen only; no stable performance claim",
            "repeat": "Only this plan/current batch; every planned candidate has six rounds; correct candidate median<fixedCSV is a win",
            "initial39": "Initial12 observed losses plus27 historical-pending shapes; split cohorts avoid counting historical evidence as current wins",
            "previous18": "Frozen selection receipt prior_current data is only a comparison reference; never fills current missing rounds"}}
    if args.require_complete and (summary["pending"] or known_missing):
        raise ValueError("Require-complete failed: current shapes/known candidates are pending, missing, or incorrect")
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    dump(out / "report.json", {"summary": summary, "shapes": comparisons, "directions": directions,
                               "known_missing": known_missing, "remaining_losses": losses})
    dump(out / "summary.json", summary)
    for name, rows in (("shape_status", comparisons), ("prior18_changes", prior18), ("new38_validation", extra38),
                       ("initial39_status", old39), ("remaining_losses", losses), ("direction_comparisons", directions), ("known_missing", known_missing)):
        fields = tuple(dict.fromkeys(key for row in rows for key in row)) or FIELDS
        accounting.atomic_csv(out / (name + ".csv"), rows, fields)
    caption = "初筛（单轮，不算稳定确认）" if args.phase == "screen" else "本次六轮中位数确认"
    lines = [f"{caption}：完成{summary['completed']}/{summary['expected']}，胜CSV {summary['wins_csv']}，落后{summary['losses_csv']}，无正确候选{summary['no_correct_opus']}，待测{summary['pending']}。", "",
        f"初始旧落后集合{len(old39)}个，本轮胜CSV {summary['initial39']['wins_csv']}；此前已测{len(prior18)}个，本轮胜CSV {summary['previous18']['wins_csv']}；新增当前验证{len(extra38)}个，本轮胜CSV {summary['new38']['wins_csv']}。", "",
        "CSV基线按gfx/cu_num/M/N/K/w_scale_block六字段匹配。父配置未测时不混用父单轮与新配置中位数；历史结果只作变化参照。", ""]
    lines += [f"{group}剩余M：{values}" for group, values in summary["remaining_M_by_NK"].items() if values]
    accounting.atomic_text(out / "summary_zh.md", "\n".join(lines) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key not in ("batch", "definitions")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
