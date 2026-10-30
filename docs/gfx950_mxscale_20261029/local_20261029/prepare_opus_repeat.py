#!/usr/bin/env python3
"""CPU-only OPUS CO/JIT shortlist and exact historical-winner mapping.

Default: step2_profile.csv + regressions_46_shapes.csv -> step2_paired/.
--audit-only writes remote mapping before the profile is available.
The remote winner is all_shapes_421.csv's opus_kernel/opus_us, not old_kernel.
"""

import argparse
import ast
import csv
import importlib.util
import json
import math
import re
import sys
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

LOCAL = Path(__file__).resolve().parent
ROOT = LOCAL.parents[2]
DOC = ROOT / "docs/gfx950_mxscale_20261029"
DATA = DOC / "handoff_data"
KEYS = ("gfx", "cu_num", "M", "N", "K", "w_scale_block")
NAME_ROOT = "opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_"


def csv_rows(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        return [{str(k).strip(): v.strip() if isinstance(v, str) else v
                 for k, v in row.items() if k and not k.startswith("Unnamed:")}
                for row in csv.DictReader(stream)]


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


@lru_cache(maxsize=1)
def pure_common():
    path = ROOT / "csrc/opus_gemm/opus_gemm_common.py"
    spec = importlib.util.spec_from_file_location("_opus_repeat_common", path)
    common = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = common
    spec.loader.exec_module(common)
    return common


@lru_cache(maxsize=1)
def pure_full_name_function():
    """Use the tuner's pure proven-complete aliases without importing aiter."""
    path = ROOT / "csrc/gemm_a8w8_mxfp8_bpreshuffle/gemm_a8w8_mxfp8_bpreshuffle_tune.py"
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_opus_jit_full_name"]
    if len(nodes) != 1:
        raise ValueError("Cannot locate the tuner's pure complete-name alias function")
    scope = {"re": re}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    return scope["_opus_jit_full_name"]


def traits_args(traits):
    match = re.fullmatch(r"\s*(\w+)\s*<([^<>]*)>\s*", traits or "")
    if not match:
        return None
    args = tuple(value.strip().lower() for value in match[2].split(",") if value.strip())
    if any(not re.fullmatch(r"-?\d+|true|false", value) for value in args):
        return None
    return match[1], args


def numeric_full_name(geometry, args):
    pipeline, bm, bn, bk, tm, tn = geometry
    return f"{NAME_ROOT}{pipeline}_{bm}x{bn}x{bk}_w{tm}x{tn}_" + "_".join(args)


def registry_metadata(family="a8w8_blockscale_bpreshuffle", gfx="gfx950", cu_num=256):
    """Return (metadata, exact_guard_pairs), without torch/aiter/GPU imports.

    metadata[kid] has kid, kernelName, traits (stable string), geometry, bounded,
    kind, is_co, bpreshuffle_co (CO symbol or None), and full_name_aliases.
    CO traits include body specialization, so runner identity checks preserve
    output/addressing choices which the geometry alone cannot prove.
    """
    registry = pure_common().get_registry(family, gfx, int(cu_num))
    metadata, guard_identities = {}, defaultdict(list)
    for kid, instance in registry.items():
        is_co = instance.bpreshuffle_co is not None
        entry = instance.bpreshuffle_co or instance.bpreshuffle_entry
        if not entry:
            raise ValueError(f"Unsupported OPUS metadata for kid {kid}")
        geometry = (entry["pipeline"], instance.B_M, instance.B_N,
                    instance.B_K, instance.T_M, instance.T_N)
        bounded = None
        aliases = []
        if is_co:
            traits = json.dumps({name: entry.get(name) for name in
                                 ("traits", "body_func", "body_args", "pad_m", "wide_output",
                                  "fixed_k", "grid_layout", "launch_bounds", "device_flags")},
                                sort_keys=True, separators=(",", ":"))
        else:
            traits = entry["traits"]
            parsed = traits_args(traits)
            if entry["pipeline"] == "pingpong_lds":
                if parsed is None or parsed[0] != "opus_gemm_bpreshuffle_pingpong_lds_traits" or len(parsed[1]) != 13:
                    raise ValueError(f"Expected current 13-parameter m-repeat traits for kid {kid}")
                bounded = parsed[1][10] == "true"
                aliases.append(numeric_full_name(geometry, parsed[1]))
                guard_identities[(geometry, parsed[1])].append(int(kid))
            else:
                alias = pure_full_name_function()(instance)
                if alias:
                    aliases.append(alias)
        metadata[int(kid)] = {"kid": int(kid), "kernelName": instance.name,
                              "traits": traits, "geometry": list(geometry), "bounded": bounded,
                              "kind": "CO" if is_co else "JIT", "is_co": is_co,
                              "bpreshuffle_co": entry.get("symbol", instance.name) if is_co else None,
                              "full_name_aliases": sorted(set(aliases))}
    pairs = {kid: [] for kid in metadata}
    for kid, meta in metadata.items():
        if meta["bounded"] is None:
            continue
        args = list(traits_args(meta["traits"])[1])
        args[10] = "false" if meta["bounded"] else "true"
        pairs[kid] = guard_identities.get((tuple(meta["geometry"]), tuple(args)), [])
    return metadata, pairs


def normalized_historical_name(name):
    """Remove only the three explicitly false obsolete m-repeat pin slots."""
    match = re.fullmatch(re.escape(NAME_ROOT) + r"pingpong_lds_(\d+)x(\d+)x(\d+)_w(\d+)x(\d+)_(.+)", name)
    if not match:
        return name, "unchanged"
    args = tuple(value.lower() for value in match[6].split("_"))
    if len(args) != 16 or any(not re.fullmatch(r"-?\d+|true|false", value) for value in args):
        return name, "unchanged"
    if args[9:12] != ("false", "false", "false"):
        return None, "obsolete_mrepeat_pin_values_nonfalse"
    geometry = ("pingpong_lds", *(int(match[index]) for index in range(1, 6)))
    return numeric_full_name(geometry, args[:9] + args[12:]), "removed_false_mrepeat_pin_slots"


def name_indexes(metadata):
    exact, aliases = defaultdict(set), defaultdict(set)
    for kid, meta in metadata.items():
        exact[meta["kernelName"]].add(kid)
        for name in meta["full_name_aliases"]:
            aliases[name].add(kid)
    return exact, aliases


def map_remote(name, metadata, indexes, shape):
    exact, aliases = indexes
    normalized, normalization = normalized_historical_name(name)
    if name in exact:
        matched, method = exact[name], "exact_current_name"
    elif normalized is not None and normalized in aliases:
        matched = aliases[normalized]
        method = "complete_jit_traits_after_false_pin_removal" if normalization != "unchanged" else "complete_jit_traits"
    else:
        matched, method = set(), "unmapped_complete_name"
    registry = pure_common().get_registry("a8w8_blockscale_bpreshuffle", shape[0], shape[1])
    legal = sorted(kid for kid in matched if pure_common().a8w8_mxscale_bpreshuffle_supports_shape(registry[kid], *shape[2:5]))
    return {"method": method, "normalization": normalization, "normalized_name": normalized,
            "name_matches": sorted(matched), "legal_current_kids": legal,
            "mapped": bool(legal),
            "reason": "matched_complete_name_and_shape" if legal else "name_matched_but_shape_illegal" if matched
            else normalization if normalized is None else "no_complete_current_name_or_traits_alias"}


def observation(records, kid, metadata, err_limit):
    rows = records.get(kid, [])
    record = rows[-1] if rows else None
    latency, error = (number(record.get("us")), number(record.get("errRatio"))) if record else (None, None)
    correct = latency is not None and latency > 0 and error is not None and 0 <= error <= err_limit
    status = "missing" if record is None else "invalid_time" if latency is None or latency <= 0 else "correct" if correct else "incorrect"
    return {"kid": kid, "kernelName": metadata[kid]["kernelName"], "us": latency,
            "errRatio": error, "correct": correct, "status": status, "observations": len(rows)}


def write_csv(path, rows):
    columns = list(dict.fromkeys(column for row in rows for column in row))
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=LOCAL / "step2_profile.csv")
    parser.add_argument("--shapes", type=Path, default=DATA / "regressions_46_shapes.csv")
    parser.add_argument("--baseline", type=Path, default=DATA / "csv_baseline.csv")
    parser.add_argument("--history", type=Path, default=DOC / "all_shapes_421.csv")
    parser.add_argument("--output-dir", type=Path, default=LOCAL / "step2_paired")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--err-ratio", type=float, default=0.05)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if args.topk < 1 or not 0 <= args.err_ratio <= 1:
        parser.error("topk must be positive and err-ratio in [0,1]")
    shapes = sorted({key(row) for row in csv_rows(args.shapes)})
    if len({shape[:2] for shape in shapes}) != 1 or not shapes or shapes[0][0] != "gfx950":
        parser.error("One gfx950/CU target per plan is required")
    metadata, pairs = registry_metadata(gfx=shapes[0][0], cu_num=shapes[0][1])
    indexes = name_indexes(metadata)
    history = {key(row): row for row in csv_rows(args.history)}
    baseline = {key(row): float(row["us"]) for row in csv_rows(args.baseline)}
    mappings = {}
    for shape in shapes:
        if shape not in history:
            parser.error(f"No historical record for {shape}")
        mappings[shape] = map_remote(history[shape]["opus_kernel"], metadata, indexes, shape)
    observed = defaultdict(lambda: defaultdict(list))
    if not args.audit_only:
        if not args.profile.exists():
            parser.error(f"Profile not yet available: {args.profile}; use --audit-only for name coverage")
        for row in csv_rows(args.profile):
            if row.get("libtype") != "opus":
                continue
            shape, kid = key(row), int(row["kernelId"])
            if shape not in mappings:
                continue
            if kid not in metadata:
                parser.error(f"Profile contains unavailable current kid {kid}")
            if number(row.get("us")) is not None and float(row["us"]) > 0 and row.get("kernelName") != metadata[kid]["kernelName"]:
                parser.error(f"Profile name differs from current kid {kid} identity")
            observed[shape][kid].append(row)
    mapped_rows, shape_rows, candidate_rows, plan_shapes = [], [], [], []
    for shape in shapes:
        old = history[shape]
        mapping = mappings[shape]
        csv_us = baseline.get(shape)
        old_us = number(old.get("opus_us"))
        old_is_co = str(old.get("best_opus_is_co", "")).lower() == "true"
        results = {kid: observation(observed[shape], kid, metadata, args.err_ratio) for kid in observed[shape]}
        valid = sorted((result for result in results.values() if result["correct"]), key=lambda result: result["us"])
        new_best = valid[0] if valid else None
        best_co = next((result for result in valid if metadata[result["kid"]]["kind"] == "CO"), None)
        mapped_results = [observation(observed[shape], kid, metadata, args.err_ratio) for kid in mapping["legal_current_kids"]]
        valid_mapped = sorted((result for result in mapped_results if result["correct"]), key=lambda result: result["us"])
        old_now_best = valid_mapped[0] if valid_mapped else None
        shape_info = {**shape_row(shape), "csv_baseline_us": csv_us,
                      "remote_winner_name": old["opus_kernel"], "remote_reference_us": old_us,
                      "remote_kind": "CO" if old_is_co else "JIT", "mapping_method": mapping["method"],
                      "mapping_reason": mapping["reason"], "mapped": mapping["mapped"],
                      "mapped_current_kids": json.dumps(mapping["legal_current_kids"]),
                      "mapped_count": len(mapping["legal_current_kids"]),
                      "mapped_current_kinds": json.dumps(sorted({metadata[kid]["kind"] for kid in mapping["legal_current_kids"]})),
                      "remote_wins_csv_historically": old_us is not None and csv_us is not None and old_us < csv_us,
                      "current_remote_correct": old_now_best is not None,
                      "current_remote_best_kid": old_now_best["kid"] if old_now_best else None,
                      "current_remote_best_us": old_now_best["us"] if old_now_best else None,
                      "current_remote_wins_csv": bool(old_now_best and csv_us is not None and old_now_best["us"] < csv_us),
                      "new_best_kid": new_best["kid"] if new_best else None,
                      "new_best_name": new_best["kernelName"] if new_best else None,
                      "new_best_us": new_best["us"] if new_best else None,
                      "new_best_wins_csv": bool(new_best and csv_us is not None and new_best["us"] < csv_us),
                      "best_co_kid": best_co["kid"] if best_co else None,
                      "best_co_us": best_co["us"] if best_co else None,
                      "best_co_wins_csv": bool(best_co and csv_us is not None and best_co["us"] < csv_us),
                      "new_best_over_remote_current_speedup": old_now_best["us"] / new_best["us"] if old_now_best and new_best else None,
                      "profile_candidates": len(results), "correct_profile_candidates": len(valid)}
        shape_rows.append(shape_info)
        for result in mapped_results or [None]:
            mapped_rows.append({**shape_row(shape), "remote_winner_name": old["opus_kernel"],
                                "remote_reference_us": old_us, "remote_kind": "CO" if old_is_co else "JIT",
                                "mapping_method": mapping["method"], "mapping_reason": mapping["reason"],
                                "normalized_remote_name": mapping["normalized_name"],
                                "mapped": result is not None, "current_kid": result["kid"] if result else None,
                                "current_name": result["kernelName"] if result else None,
                                "current_kind": metadata[result["kid"]]["kind"] if result else None,
                                "initial_status": result["status"] if result else "unmapped",
                                "initial_correct": bool(result and result["correct"]),
                                "initial_us": result["us"] if result else None,
                                "initial_errRatio": result["errRatio"] if result else None,
                                "csv_baseline_us": csv_us,
                                "initial_wins_csv": bool(result and result["correct"] and csv_us is not None and result["us"] < csv_us)})
        if args.audit_only:
            continue
        selected = {result["kid"] for result in valid[:args.topk]} | set(mapping["legal_current_kids"])
        if best_co:
            selected.add(best_co["kid"])
        ordered = sorted(selected, key=lambda kid: (not results.get(kid, {}).get("correct", False),
                                                  results.get(kid, {}).get("us") or math.inf, kid))
        plan_candidates = []
        for kid in ordered:
            result = observation(observed[shape], kid, metadata, args.err_ratio)
            roles = []
            if kid in {r["kid"] for r in valid[:args.topk]}:
                roles.append("top3" if args.topk == 3 else "topk")
            if kid in mapping["legal_current_kids"]:
                roles.append("remote_exact")
            if best_co and kid == best_co["kid"]:
                roles.append("best_co")
            candidate = {**metadata[kid], **result, "exact_opposite_kids": pairs[kid], "roles": roles,
                         "remote_reference_us": old_us if "remote_exact" in roles else None}
            plan_candidates.append(candidate)
            candidate_rows.append({**shape_row(shape), "kid": kid, "kernelName": candidate["kernelName"],
                                   "kind": candidate["kind"], "roles": json.dumps(roles), "initial_status": result["status"],
                                   "initial_correct": result["correct"], "initial_us": result["us"],
                                   "initial_errRatio": result["errRatio"], "csv_baseline_us": csv_us,
                                   "initial_wins_csv": bool(result["correct"] and csv_us is not None and result["us"] < csv_us),
                                   "geometry": json.dumps(candidate["geometry"])})
        plan_shapes.append({"shape": shape_row(shape), "csv_baseline_us": csv_us,
                            "remote_winner": {"kernelName": old["opus_kernel"], "us": old_us, "is_co": old_is_co,
                                              "mapping": mapping}, "candidates": plan_candidates})
    summary = {"expected_shapes": len(shapes), "historical_source": str(args.history),
               "historical_name_column": "opus_kernel", "historical_time_column": "opus_us",
               "profile": str(args.profile) if not args.audit_only else None, "audit_only": args.audit_only,
               "historical_kinds": dict(Counter(row["remote_kind"] for row in shape_rows)),
               "mapped_shapes": sum(row["mapped"] for row in shape_rows),
               "mapped_co_winner_shapes": sum(row["mapped"] and row["remote_kind"] == "CO" for row in shape_rows),
               "mapped_jit_winner_shapes": sum(row["mapped"] and row["remote_kind"] == "JIT" for row in shape_rows),
               "mapping_methods": dict(Counter(row["mapping_method"] for row in shape_rows)),
               "unmapped_shapes": sum(not row["mapped"] for row in shape_rows),
               "profile_complete_shapes": sum(row["profile_candidates"] > 0 for row in shape_rows),
               "new_best_wins_csv": sum(row["new_best_wins_csv"] for row in shape_rows),
               "mapped_old_winner_correct_shapes": sum(row["current_remote_correct"] for row in shape_rows),
               "mapped_old_winner_wins_csv": sum(row["current_remote_wins_csv"] for row in shape_rows),
               "repeat_shapes": len(plan_shapes), "repeat_candidates": sum(len(item["candidates"]) for item in plan_shapes),
               "selection": f"Latest profile per current shape/kid; fastest correct Top{args.topk}, best correct CO, and every legal complete-name remote match",
               "identity_policy": "CO full symbol only; JIT full traits alias; m-repeat removes three explicitly false obsolete pin slots. Historical kid ignored.",
               "torch_imported": "torch" in sys.modules}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "remote_mapping.csv", mapped_rows)
    write_csv(args.output_dir / "analysis_by_shape.csv", shape_rows)
    if not args.audit_only:
        write_csv(args.output_dir / "shortlist.csv", candidate_rows)
        plan = {"version": 2, "family": "a8w8_blockscale_bpreshuffle", "source_profile": str(args.profile),
                "metadata_module": "prepare_opus_repeat", "correctness_limit": args.err_ratio,
                "selection": summary["selection"], "shapes": plan_shapes}
        (args.output_dir / "repeat_plan.json").write_text(json.dumps(plan, indent=2, allow_nan=False) + "\n")
    (args.output_dir / "analysis.json").write_text(json.dumps({"summary": summary, "shapes": shape_rows,
                                                               "remote_mapping": mapped_rows}, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
