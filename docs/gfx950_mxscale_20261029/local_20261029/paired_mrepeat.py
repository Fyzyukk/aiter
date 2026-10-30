#!/usr/bin/env python3
"""CPU-only m-repeat guard pairing and repeat-plan generation.

Defaults read step1_profile.csv and write step1_paired/*.csv plus repeat_plan.json.
Only the standard library is imported; current C++ traits are the identity.
"""

import argparse
import csv
import importlib.util
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
LOCAL = Path(__file__).resolve().parent
DATA = ROOT / "docs/gfx950_mxscale_20261029/handoff_data"
KEYS = ("gfx", "cu_num", "M", "N", "K", "w_scale_block")
TRAITS_FIELDS = ("BM", "BN", "ScalePanel", "FixedK", "WaveM", "GroupM",
                 "CacheA", "CacheB", "UseFragments", "XcdOrder", "BoundedOutput",
                 "PublishM", "GroupN")
TARGET_KIDS = tuple(range(48198, 48212))


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        return [{str(k).strip(): v.strip() if isinstance(v, str) else v
                 for k, v in row.items() if k and not k.startswith("Unnamed:")}
                for row in csv.DictReader(stream)]


def shape_key(row):
    return (str(row.get("gfx", "gfx950")), int(row.get("cu_num", 256)),
            int(row["M"]), int(row["N"]), int(row["K"]),
            str(row.get("w_scale_block", "128x128")))


def finite_number(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def registry_metadata():
    """Load the pure registry directly, bypassing the aiter package entirely."""
    path = ROOT / "csrc/opus_gemm/opus_gemm_common.py"
    spec = importlib.util.spec_from_file_location("_paired_opus_common", path)
    common = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = common
    spec.loader.exec_module(common)
    metadata = {}
    for kid, instance in common.a8w8_blockscale_mxscale_gemm_bpreshuffle_kernels_list.items():
        entry = instance.bpreshuffle_entry
        if not entry or entry.get("pipeline") != "pingpong_lds":
            continue
        match = re.fullmatch(r"opus_gemm_bpreshuffle_pingpong_lds_traits<([^<>]+)>", entry["traits"])
        if not match:
            raise ValueError(f"Unrecognized m-repeat traits for kid {kid}")
        traits = tuple(arg.strip().lower() for arg in match[1].split(","))
        if len(traits) != 13 or traits[10] not in ("true", "false"):
            raise ValueError(f"Expected current 13-parameter traits for kid {kid}")
        metadata[int(kid)] = {
            "kid": int(kid), "kernelName": instance.name, "traits": traits,
            "geometry": (entry["pipeline"], instance.B_M, instance.B_N,
                         instance.B_K, instance.T_M, instance.T_N),
            "bounded": traits[10] == "true",
        }
    by_traits = defaultdict(list)
    for kid, meta in metadata.items():
        by_traits[(meta["geometry"], meta["traits"])].append(kid)
    pairs = {}
    for kid, meta in metadata.items():
        opposite = list(meta["traits"])
        opposite[10] = "false" if meta["bounded"] else "true"
        pairs[kid] = by_traits.get((meta["geometry"], tuple(opposite)), [])
    for kid in TARGET_KIDS:
        if kid not in metadata or metadata[kid]["bounded"] or len(pairs[kid]) != 1:
            raise ValueError(f"Target kid {kid} no longer has one exact true partner")
    return metadata, pairs


def candidate_result(rows, kid, err_limit, meta):
    records = rows.get(kid, [])
    # Profile appends on resume: use the latest observation, never cherry-pick a run.
    record = records[-1] if records else None
    latency = finite_number(record.get("us")) if record else None
    error = finite_number(record.get("errRatio")) if record else None
    correct = latency is not None and latency > 0 and error is not None and 0 <= error <= err_limit
    status = ("missing" if record is None else "invalid_time" if latency is None or latency <= 0
              else "invalid_error" if error is None or error < 0 else "correct" if correct else "incorrect")
    return {"kid": kid, "kernelName": meta["kernelName"], "status": status,
            "correct": correct, "us": latency, "errRatio": error,
            "observations": len(records), "bounded": meta["bounded"]}


def best(results):
    valid = [result for result in results if result["correct"]]
    return min(valid, key=lambda result: result["us"]) if valid else None


def prefixed(result, prefix, baseline):
    return {prefix + "_" + name: value for name, value in {
        "kid": result["kid"] if result else None,
        "status": result["status"] if result else "missing",
        "correct": bool(result and result["correct"]),
        "us": result["us"] if result else None,
        "errRatio": result["errRatio"] if result else None,
        "wins_csv": bool(result and result["correct"] and baseline is not None and result["us"] < baseline),
        "csv_speedup": baseline / result["us"] if result and result["correct"] and baseline is not None else None,
    }.items()}


def write_csv(path, rows, fields=None):
    fields = fields or list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=LOCAL / "step1_profile.csv")
    parser.add_argument("--shapes", type=Path, default=DATA / "m_repeat_32_shapes.csv")
    parser.add_argument("--geometries", type=Path, default=DATA / "m_repeat_32_geometries.csv")
    parser.add_argument("--baseline", type=Path, default=DATA / "csv_baseline.csv")
    parser.add_argument("--output-dir", type=Path, default=LOCAL / "step1_paired")
    parser.add_argument("--err-ratio", type=float, default=0.05)
    parser.add_argument("--topk", type=int, default=3, help="Fastest correct m-repeat candidates per shape in repeat plan")
    parser.add_argument("--metadata-only", action="store_true", help="Write/verify the 14 exact pairs before profile exists")
    args = parser.parse_args()
    if not 0 <= args.err_ratio <= 1 or args.topk < 1:
        parser.error("err-ratio must be in [0,1] and topk must be positive")
    metadata, pairs = registry_metadata()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pair_meta = []
    for kid in TARGET_KIDS:
        meta = metadata[kid]
        row = {"false_kid": kid, "true_kid": pairs[kid][0],
               "false_name": meta["kernelName"], "true_name": metadata[pairs[kid][0]]["kernelName"]}
        row.update(dict(zip(TRAITS_FIELDS, meta["traits"])))
        pair_meta.append(row)
    write_csv(args.output_dir / "pair_metadata.csv", pair_meta)
    if args.metadata_only:
        print(json.dumps({"pairs_verified": len(pair_meta), "torch_imported": "torch" in sys.modules,
                          "output": str(args.output_dir / "pair_metadata.csv")}, indent=2))
        return
    if not args.profile.exists():
        parser.error(f"Profile is not available yet: {args.profile}")
    expected = sorted({shape_key(row) for row in read_csv(args.shapes)})
    baseline = {shape_key(row): float(row["us"]) for row in read_csv(args.baseline)}
    requested_geometries = defaultdict(set)
    for row in read_csv(args.geometries):
        requested_geometries[(int(row["M"]), int(row["N"]), int(row["K"]))].add(
            (row["pipeline"], *(int(row[field]) for field in ("BM", "BN", "BK", "TM", "TN"))))
    observed = defaultdict(lambda: defaultdict(list))
    for row in read_csv(args.profile):
        if row.get("libtype") != "opus":
            continue
        kid = int(row["kernelId"])
        if kid in metadata:
            latency = finite_number(row.get("us"))
            if latency is not None and latency > 0 and row.get("kernelName") != metadata[kid]["kernelName"]:
                parser.error(f"Profile identity differs from current registry for kid {kid}")
            observed[shape_key(row)][kid].append(row)
    pair_rows, geometry_rows, summary_rows, plan_shapes = [], [], [], []
    for shape in expected:
        fields = dict(zip(KEYS, shape))
        csv_us = baseline.get(shape)
        records = observed.get(shape, {})
        results = {kid: candidate_result(records, kid, args.err_ratio, metadata[kid]) for kid in records}
        geometries = sorted(requested_geometries.get(shape[2:5], set()) |
                            {metadata[kid]["geometry"] for kid in records})
        for kid in TARGET_KIDS:
            meta = metadata[kid]
            if meta["geometry"] not in geometries:
                continue
            opposite = pairs[kid][0]
            left = candidate_result(records, kid, args.err_ratio, meta)
            right = candidate_result(records, opposite, args.err_ratio, metadata[opposite])
            row = {**fields, "csv_baseline_us": csv_us, "geometry": json.dumps(meta["geometry"])}
            row.update(prefixed(left, "false", csv_us))
            row.update(prefixed(right, "true", csv_us))
            row.update(false_vs_true_pct=(right["us"] / left["us"] - 1) * 100
                       if left["correct"] and right["correct"] else None)
            pair_rows.append(row)
        for geometry in geometries:
            geo_results = [result for kid, result in results.items() if metadata[kid]["geometry"] == geometry]
            left = best(result for result in geo_results if not result["bounded"])
            right = best(result for result in geo_results if result["bounded"])
            row = {**fields, "geometry": json.dumps(geometry), "csv_baseline_us": csv_us,
                   "false_tested": sum(not r["bounded"] for r in geo_results),
                   "false_correct_count": sum(not r["bounded"] and r["correct"] for r in geo_results),
                   "true_tested": sum(r["bounded"] for r in geo_results),
                   "true_correct_count": sum(r["bounded"] and r["correct"] for r in geo_results)}
            row.update(prefixed(left, "best_false", csv_us))
            row.update(prefixed(right, "best_true", csv_us))
            row["best_false_vs_true_pct"] = ((right["us"] / left["us"] - 1) * 100
                                              if left and right else None)
            geometry_rows.append(row)
        left = best(result for result in results.values() if not result["bounded"])
        right = best(result for result in results.values() if result["bounded"])
        winner = best(results.values())
        summary_row = {**fields, "csv_baseline_us": csv_us,
                       "tested_mrepeat_candidates": len(results), "correct_mrepeat_candidates": sum(r["correct"] for r in results.values()),
                       "geometries": len(geometries)}
        for prefix, result in (("best_false", left), ("best_true", right), ("best_mrepeat", winner)):
            summary_row.update(prefixed(result, prefix, csv_us))
        summary_rows.append(summary_row)
        selected = {result["kid"] for result in sorted((r for r in results.values() if r["correct"]),
                                                       key=lambda r: r["us"])[:args.topk]}
        selected.update(result["kid"] for result in (left, right) if result)
        # Add exact guard partners for the shortlist when both were measured correctly.
        for kid in tuple(selected):
            selected.update(partner for partner in pairs[kid] if partner in results and results[partner]["correct"])
        # Verify all 14 restored remote guard variants where the initial full
        # search tested them, including incorrect candidates and their partners.
        # A failed initial measurement must remain visible in the repeat audit.
        for kid in TARGET_KIDS:
            if kid in results:
                selected.add(kid)
                selected.update(pairs[kid])
        selected_results = sorted((results.get(kid) or candidate_result(records, kid, args.err_ratio, metadata[kid])
                                   for kid in selected),
                                  key=lambda r: (not r["correct"], r["us"] if r["us"] is not None and r["us"] > 0 else math.inf))
        if selected_results:
            plan_shapes.append({"shape": fields, "csv_baseline_us": csv_us,
                                "candidates": [{**result, "traits": list(metadata[result["kid"]]["traits"]),
                                                "geometry": list(metadata[result["kid"]]["geometry"]),
                                                "exact_opposite_kids": pairs[result["kid"]]}
                                               for result in selected_results]})
    write_csv(args.output_dir / "pairs_by_shape.csv", pair_rows)
    write_csv(args.output_dir / "geometry_best_by_shape.csv", geometry_rows)
    write_csv(args.output_dir / "summary_by_shape.csv", summary_rows)
    summary = {"profile": str(args.profile), "selection": "latest observation per shape/kid",
               "correctness_limit": args.err_ratio, "expected_shapes": len(expected),
               "observed_mrepeat_shapes": sum(bool(observed.get(shape)) for shape in expected),
               "pairs_compared": len(pair_rows),
               "both_correct_pairs": sum(r["false_correct"] and r["true_correct"] for r in pair_rows),
               "false_faster_exact_pairs": sum(r["false_vs_true_pct"] is not None and r["false_vs_true_pct"] > 0 for r in pair_rows),
               "exact_pair_statuses": dict(Counter(f'{r["false_status"]}/{r["true_status"]}' for r in pair_rows)),
               "best_false_csv_wins": sum(r["best_false_wins_csv"] for r in summary_rows),
               "best_true_csv_wins": sum(r["best_true_wins_csv"] for r in summary_rows),
               "best_mrepeat_csv_wins": sum(r["best_mrepeat_wins_csv"] for r in summary_rows),
               "repeat_shapes": len(plan_shapes), "repeat_candidates": sum(len(r["candidates"]) for r in plan_shapes),
               "percentage_definition": "false_vs_true_pct = (true_us / false_us - 1) * 100; positive favors false",
               "geometry_comparison": "Best false/true may also differ in other traits; only pairs_by_shape isolates the guard"}
    plan = {"version": 1, "source_profile": str(args.profile), "correctness_limit": args.err_ratio,
            "selection": "topk plus best false/true, their correct partners, and every measured target kid 48198-48211 with its exact true partner", "shapes": plan_shapes}
    (args.output_dir / "repeat_plan.json").write_text(json.dumps(plan, indent=2, allow_nan=False) + "\n")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
