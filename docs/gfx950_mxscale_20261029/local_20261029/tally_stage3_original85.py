#!/usr/bin/env python3
"""CPU-only accounting for the frozen 16 observed + 69 historical losses.

The latest focused batch replaces every older record for its shapes. Outside
that batch, use only saved complete six-round candidates and keep the latest
usable record for each candidate. Historical reference timings never count as
current-machine measurements.
"""

import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import finish_workflow as accounting
import prepare_opus_repeat as metadata_helper
from prepare_co_experiment import LOCAL, ROOT, load_common


REPORTS = (
    "step1_report/report.json",
    "step2_report/report.json",
    "step3_existing_round1/repeat_report/report.json",
    "co_round1/repeat_report/report.json",
    "co_round2/repeat_report/report.json",
)


def summarize(rows):
    counts = Counter(row["status"] for row in rows)
    return {
        "expected": len(rows),
        "current_machine_six_round_confirmed": counts["wins_csv"] + counts["slower_than_csv"],
        "wins_csv": counts["wins_csv"],
        "losses_csv": counts["slower_than_csv"],
        "pending": counts["pending"],
    }


def main():
    sources = {}

    def read_json(relative):
        path = LOCAL / relative
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())

    def read_shapes(relative):
        path = LOCAL / relative
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        rows = metadata_helper.csv_rows(path)
        keys = {accounting.strict_key(row, path) for row in rows}
        if len(keys) != len(rows):
            raise ValueError("Duplicate frozen shape")
        return keys

    observed = read_shapes("step3_inputs/observed_losses_shapes.csv")
    historical = read_shapes("step3_inputs/historical_pending_losses_shapes.csv")
    if len(observed) != 16 or len(historical) != 69 or observed & historical:
        raise ValueError("Frozen initial85 must be 16 observed + 69 disjoint historical")
    expected = observed | historical
    baseline_path = metadata_helper.DATA / "csv_baseline.csv"
    sources[str(baseline_path)] = hashlib.sha256(baseline_path.read_bytes()).hexdigest()
    baseline = {}
    for row in metadata_helper.csv_rows(baseline_path):
        key = accounting.strict_key(row, baseline_path, require_scope=False)
        if key not in expected:
            continue
        if key in baseline or accounting.finite(row["us"]) is None or float(row["us"]) <= 0:
            raise ValueError("Duplicate/invalid fixed CSV baseline")
        baseline[key] = float(row["us"])
    if set(baseline) != expected:
        raise ValueError("Missing original85 fixed baseline")

    focused = read_json("focused_result_analysis/repeat/report.json")
    current = {accounting.strict_key(row, "focused strict report"): row for row in focused["shapes"]}
    if len(current) != 56 or focused["summary"]["pending"] or focused["summary"]["known_candidate_failures_or_pending"]:
        raise ValueError("Require complete strict focused56 with all known candidates correct")
    if len(expected & current.keys()) != 39:
        raise ValueError("Focused56 must replace exactly39 original85 shapes")

    common = load_common(ROOT, label="original85_tally")
    metadata_helper.pure_common = lambda: common
    metadata, _ = metadata_helper.registry_metadata()
    registry = common.get_registry("a8w8_blockscale_bpreshuffle", "gfx950", 256)
    latest_by_candidate = {}
    for relative in REPORTS:
        report = read_json(relative)
        if report["summary"]["rounds"] != 6:
            raise ValueError("Only saved six-round reports allowed")
        for row in report["candidates"]:
            key = accounting.strict_key(row, relative)
            if key not in expected or key in current:
                continue
            if not row.get("all_rounds_correct") or not row.get("timing_matches_tuner"):
                continue
            kid = int(row["kid"])
            if kid not in metadata:
                continue  # Retired experimental configurations cannot win publicly.
            if row["kernelName"] != metadata[kid]["kernelName"]:
                raise ValueError("Saved/current candidate identity differs")
            if not common.a8w8_mxscale_bpreshuffle_supports_shape(registry[kid], *key[2:5]):
                raise ValueError("Saved public candidate is shape-illegal")
            if row["csv_baseline_us"] != baseline[key] or row["rounds_written"] != 6 or row["correct_rounds"] != 6:
                raise ValueError("Saved candidate baseline/round completeness differs")
            if not math.isfinite(row["median_us"]) or row["median_us"] <= 0:
                raise ValueError("Invalid saved candidate median")
            latest_by_candidate[key, kid] = {**row, "source_report": str(LOCAL / relative)}

    results = []
    for key in sorted(expected):
        if key in current:
            row = current[key]
            if not row["repeat_complete"] or row["csv_baseline_us"] != baseline[key]:
                raise ValueError("Incomplete/different-baseline focused original85 shape")
            result = {"status": row["status"], "kid": row["kid"], "kernelName": row["kernelName"],
                      "us": row["us"], "source_report": str(LOCAL / "focused_result_analysis/repeat/report.json"),
                      "evidence_kind": "latest_focused_batch_six_round_median", "focused_current_batch": True}
        else:
            candidates = [row for (shape, _), row in latest_by_candidate.items() if shape == key]
            best = min(candidates, key=lambda row: (row["median_us"], row["kid"])) if candidates else None
            result = {"status": "pending" if best is None else "wins_csv" if best["median_us"] < baseline[key] else "slower_than_csv",
                      "kid": best["kid"] if best else None, "kernelName": best["kernelName"] if best else None,
                      "us": best["median_us"] if best else None, "source_report": best["source_report"] if best else None,
                      "evidence_kind": "saved_current_machine_six_round_median" if best else "historical_reference_only",
                      "focused_current_batch": False}
        results.append({**accounting.shape_row(key), "original_cohort": "observed16" if key in observed else "historical69",
                        **result, "csv_baseline_us": baseline[key],
                        "csv_gap_pct": (result["us"] / baseline[key] - 1) * 100 if result["us"] else None})
    summary = {"original85": summarize(results),
               "observed16": summarize([row for row in results if row["original_cohort"] == "observed16"]),
               "historical69": summarize([row for row in results if row["original_cohort"] == "historical69"]),
               "focused39": summarize([row for row in results if row["focused_current_batch"]]),
               "outside46": summarize([row for row in results if not row["focused_current_batch"]]),
               "gpu_used_by_tool": False,
               "policy": "Frozen original85; latest focused whole-shape replacement; outside46 latest usable complete six-round record per public candidate; fixed six-field CSV; pending is not confirmed loss"}
    out = LOCAL / "stage3_original85_status"
    out.mkdir(exist_ok=True)
    report = {"summary": summary, "source_sha256": sources, "shapes": results}
    accounting.atomic_text(out / "report.json", json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    accounting.atomic_text(out / "summary.json", json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    accounting.atomic_csv(out / "shape_status.csv", results, tuple(results[0]))
    for name, status in (("recovered_wins", "wins_csv"), ("remaining_losses", "slower_than_csv"), ("pending_shapes", "pending")):
        accounting.atomic_csv(out / (name + ".csv"), [row for row in results if row["status"] == status], tuple(results[0]))
    text = ("第三步原始待处理集合85个：16个本机已测落后 + 69个历史落后、当时尚待本机确认。\n\n"
            "当前去重结果：8个已恢复并胜CSV，35个有本机六轮证据但仍落后，42个尚未补测。"
            "共43/85有本机六轮证据；未完成的77个应拆为35个确认落后和42个待测。\n\n"
            "本次focused覆盖原集合39个，六轮结果7胜/32落后；其余46个中，"
            "已有六轮结果1胜/3落后，42个仍待完整421验证。"
            "focused形状采用本批结果，旧批的小幅胜出不覆盖本批落后。\n\n"
            "原16个本机落后：恢复4个，仍落后12个。原69个历史落后："
            "本机确认胜4个、确认落后23个、待测42个。\n")
    accounting.atomic_text(out / "summary_zh.md", text)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
