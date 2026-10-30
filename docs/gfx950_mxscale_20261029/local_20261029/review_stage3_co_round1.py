#!/usr/bin/env python3
"""CPU review of the fixed round-one CO experiment; prepare a small follow-up.

Reads saved six-round events only. Does not build, tune, integrate, or edit an
experimental manifest. The generated plan keeps runtime K and Unroll2.
"""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

LOCAL = Path(__file__).resolve().parent


def shape_key(row):
    return tuple(int(row[key]) for key in ("M", "N", "K"))


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=LOCAL / "co_round1")
    parser.add_argument("--existing-report", type=Path,
                        default=LOCAL / "step3_existing_round1/repeat_report/report.json")
    parser.add_argument("--additional-report", type=Path, action="append", default=[])
    parser.add_argument("--review-only", action="store_true",
                        help="Review another completed CO round without writing the round-two plan")
    parser.add_argument("--output-dir", type=Path,
                        default=LOCAL / "co_round1/independent_review")
    args = parser.parse_args()
    report_path = args.run_dir / "repeat_report/report.json"
    report = json.loads(report_path.read_text())
    existing = json.loads(args.existing_report.read_text())
    previous_reports = [existing, *(json.loads(path.read_text()) for path in args.additional_report)]
    point_file = args.run_dir / "identity_and_points.json"
    if not point_file.exists():
        point_file = args.run_dir / "receipt.json"
    points = json.loads(point_file.read_text())["points"]
    if report["summary"]["pending_shapes"] or report["summary"]["rounds"] != 6:
        raise ValueError("Review requires all six rounds for every experimental shape")
    candidates = {(*shape_key(row), row["kid"]): row for row in report["candidates"]}
    raw = {}
    for source in report["summary"]["files"]:
        for line in Path(source).read_text().splitlines():
            event = json.loads(line)
            key = (*shape_key(event["shape"]), event["kid"], event["round"])
            if key in raw:
                raise ValueError(f"Duplicate saved event {key}")
            current = candidates[key[:-1]]
            if event["kernelName"] != current["kernelName"]:
                raise ValueError("Saved kernel identity differs from the completed report")
            if (not event["correct"] or event["rotate_args"] != 0
                    or not math.isfinite(event["us"]) or event["us"] <= 0
                    or not math.isfinite(event["errRatio"]) or not 0 <= event["errRatio"] <= 0.05):
                raise ValueError("Review contains incorrect or incompatible saved timing")
            raw[key] = event
    comparisons = []
    for point in points:
        kid, parent = point["new_kid"], point["seed_kid"]
        for shape in map(tuple, point["shapes"]):
            new = candidates[(*shape, kid)]
            old = candidates[(*shape, parent)]
            events = [(raw[(*shape, kid, rnd)], raw[(*shape, parent, rnd)]) for rnd in range(6)]
            if len({event["gpu_logical_id"] for pair in events for event in pair}) != 1:
                raise ValueError("Paired rounds span different GPUs")
            order = [new_event["position"] < old_event["position"] for new_event, old_event in events]
            # The runner rotates the starting candidate between round pairs;
            # each pair (0,1), (2,3), (4,5) uses the exact reversed order.
            if any(order[rnd] == order[rnd + 1] for rnd in (0, 2, 4)):
                raise ValueError("New/parent execution order did not alternate")
            ratios = [old_event["us"] / new_event["us"] for new_event, old_event in events]
            if not math.isclose(statistics.median([event["us"] for event, _ in events]),
                                new["median_us"], rel_tol=1e-12):
                raise ValueError("New median differs from saved events")
            comparisons.append({"M": shape[0], "N": shape[1], "K": shape[2],
                "new_kid": kid, "parent_kid": parent, "axis": point["axis"],
                "after": point["after"], "new_median_us": new["median_us"],
                "parent_median_us": old["median_us"], "csv_baseline_us": new["csv_baseline_us"],
                "median_gain_pct": (old["median_us"] / new["median_us"] - 1) * 100,
                "paired_median_gain_pct": (statistics.median(ratios) - 1) * 100,
                "paired_round_gain_pct": [(ratio - 1) * 100 for ratio in ratios],
                "faster_rounds": sum(ratio > 1 for ratio in ratios),
                "new_before_parent": order, "csv_gap_pct": (new["median_us"] / new["csv_baseline_us"] - 1) * 100})
    previous = {shape_key(row): row for row in existing["shapes"]}
    shapes = []
    for current in report["shapes"]:
        key = shape_key(current)
        old = previous[key]
        choices = [(current["best_repeat_median_us"], current["best_repeat_kid"], "co_round1"),
                   (old["best_repeat_median_us"], old["best_repeat_kid"], "existing_full_geometry")]
        choices[0] = (*choices[0][:2], args.run_dir.name)
        for previous_report in previous_reports[1:]:
            for row in previous_report["shapes"]:
                if shape_key(row) == key and row["repeat_complete"]:
                    choices.append((row["best_repeat_median_us"], row["best_repeat_kid"], "additional_completed_report"))
        measured_us, measured_kid, source = min(choices)
        shapes.append({"M": key[0], "N": key[1], "K": key[2],
            "csv_baseline_us": current["csv_baseline_us"],
            "experiment_best_kid": current["best_repeat_kid"],
            "experiment_best_us": current["best_repeat_median_us"],
            "experiment_csv_gap_pct": (current["best_repeat_median_us"] / current["csv_baseline_us"] - 1) * 100,
            "existing_best_kid": old["best_repeat_kid"], "existing_best_us": old["best_repeat_median_us"],
            "best_observed_kid": measured_kid, "best_observed_us": measured_us,
            "best_observed_source": source,
            "best_observed_csv_gap_pct": (measured_us / current["csv_baseline_us"] - 1) * 100,
            "scope_note": "Past full-geometry best retained; new CO remains experimental"})
    shapes.sort(key=lambda row: row["best_observed_csv_gap_pct"])
    controls = [[6144, 6144, 7168], [6144, 7168, 16384], [8192, 7168, 7168],
                [8192, 7168, 16384], [32768, 7168, 16384]]
    long_k = [[6144, 7168, 16384], [8192, 7168, 16384]]
    experiments = []
    for seed in (100001, 100003):
        experiments.append({"seed_kid": seed, "vary": {"GroupM": 16}, "shapes": controls,
            "reason": "GM8 gives six-round paired gains on near-CSV controls; test legal GM16 only, retaining panel64, runtimeK, Unroll2 and output route. Include GM4-favored long-K controls to avoid assuming a universal grouping win."})
    for seed in (100000, 100001, 100002, 100003):
        experiments.append({"seed_kid": seed, "vary": {"Panel": 32},
            "shapes": long_k if seed in (100000, 100002) else controls,
            "reason": "Retain the measured GM4/GM8 and normal/wide route; change only SFA/SFB panel64 to panel32 (LDS152064 to143616 bytes). Compare refill frequency and temporary scale-register lifetime on near-CSV long-K controls; runtimeK and Unroll2 remain fixed."})
    narrow = [[4096, 1536, 7168]]
    pairs = [{"b_seed_kid": 100022, "overlap_seed_kid": 100023,
              "vary": {"NB": nb}, "shapes": narrow,
              "reason": f"Same-shape exact BD1 parents are legal. Test NB{nb}/BD1 with actual B_AHEAD2; keep XCDtrue, geometry, pin, caches and Pub0 fixed. NB3BD1 shortens static ring period to3; NB2BD1 reduces LDS to41984 bytes. Compare against original103/295, first-round NB3BD2, and BD1 parents."}
             for nb in (2, 3)]
    pairs.append({"b_seed_kid": 103, "overlap_seed_kid": 295,
                  "vary": {"BD": 3}, "shapes": narrow,
                  "reason": "Actual prefetch-distance probe: legal NB4BD3 changes B_AHEAD4 to6 and B_SETS5 to7 while LDS74752 and all other traits remain fixed. Compare saved BA2/4 controls; increased distance is a hypothesis, not a predicted win."})
    plan = {"source_report": str(report_path.resolve()), "source_status": "completed six-round medians",
            "budget": {"max_new_co": 12, "representative_shapes": 6},
            "experiments": experiments, "paired_experiments": pairs,
            "required_existing_winners": [{"shape": narrow[0], "kid": 48096,
                "reason": "Keep the full-geometry m-repeat JIT best59.16159us; CO-only winner60.35816us is not the overall best."}],
            "policy": "NT only. Keep all current804 identities and host restrictions. RuntimeK and Unroll2 fixed. First screen Top0 specified geometries; new/parent six-round confirmation precedes integration/full421."}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dump(args.output_dir / "review.json", {"summary": report["summary"], "comparisons": comparisons, "shapes": shapes})
    if not args.review_only:
        dump(args.output_dir / "next_round_plan.json", plan)
    for filename, rows in (("paired_new_parent.csv", comparisons), ("shape_priority.csv", shapes)):
        with (args.output_dir / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                             for key, value in row.items()} for row in rows)
    summary = report["summary"]
    lines = [f"{args.run_dir.name}：{summary['repeat_complete_shapes']}/{summary['expected_shapes']} shape、{summary['candidate_groups_correct_all_rounds']}/{summary['planned_candidates']}候选完成六轮且全正确，{summary['repeat_wins_csv_median']}胜CSV、{summary['remaining_losses']}仍落后。",
             "", "新CO对精确parent的逐轮收益见paired_new_parent.csv。窄N的CO结果须保留上一轮完整几何JIT最佳共同评估。", "",
             "| Shape | 本轮最佳ID | 本轮µs | 距CSV | 保留完整几何后最佳ID | 最佳µs | 距CSV |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in shapes:
        lines.append(f"| {row['M']}×{row['N']}×{row['K']} | {row['experiment_best_kid']} | {row['experiment_best_us']:.5f} | {row['experiment_csv_gap_pct']:.3f}% | {row['best_observed_kid']} | {row['best_observed_us']:.5f} | {row['best_observed_csv_gap_pct']:.3f}% |")
    if not args.review_only:
        lines.extend(["", "下一轮准备12个新CO、6个代表shape：2个GM16、4个同GM的Panel32、2对合法NB2/3BD1、1对BD3实际距离探针。全部保持runtimeK/Unroll2；此文件只准备CPU计划，未build或执行GPU。", "",
                      "所有固定输入、已有plan和报告保持原样；新计划需由extend工具完成CPU身份/合法性/重复核对后，再由root调度GPU0–3。"])
    else:
        lines.extend(["", "此文件只复核保存的CPU结果与逐轮事件，没有修改实验输入、manifest、公共源码或执行GPU。保存的最佳值跨轮比较仅用于保留候选；正式收益由同形同GPU的当前new/parent配对确定。"])
    (args.output_dir / "analysis_zh.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"output_dir": str(args.output_dir), "paired_shape_comparisons": len(comparisons),
        "stable_new_parent_improvements": sum(row["faster_rounds"] >= 4 and row["median_gain_pct"] > 0 for row in comparisons),
        "faster_in_all_six_rounds": sum(row["faster_rounds"] == 6 for row in comparisons),
        "new_plan_written": not args.review_only,
        "new_plan_co_count": None if args.review_only else len(experiments) + 2 * len(pairs),
        "new_plan_shape_count": None if args.review_only else 6,
        "gpu_used": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
