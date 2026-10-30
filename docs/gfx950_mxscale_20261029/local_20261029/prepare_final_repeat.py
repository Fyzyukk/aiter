#!/usr/bin/env python3
"""CPU-only final full421 shortlist for the version2 six-round GPU runner.

Keep correct global Top5, Top2 per pipeline, exact remote full-name winners,
the best complete current six-round candidate per shape, and explicitly
verified best/new-CO candidates from --current-report. Historical profiles are
not read implicitly. All selected IDs/names/traits/geometries come from the
current available registry and pass the actual runner's CPU --dry-run.
"""

import argparse
import glob
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from finish_workflow import strict_key, repeat_batch
from prepare_opus_repeat import (LOCAL, ROOT, DOC, DATA, csv_rows, number, shape_row,
                                 registry_metadata, pure_common, name_indexes, map_remote)

ROUNDS = set(range(6))
_REGISTRY_CACHE = {}


def boolean(value):
    return value is True or str(value).lower() == "true"


def valid(row, limit):
    us, err = number(row.get("us")), number(row.get("errRatio"))
    return bool(us is not None and us > 0 and err is not None and 0 <= err <= limit
                and boolean(row.get("correct", True)) and not row.get("failure"))


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expand(patterns):
    paths, seen = [], set()
    for pattern in patterns:
        matches = sorted(glob.glob(str(pattern)))
        if not matches:
            raise ValueError(f"No input matches {pattern}")
        for match in matches:
            path = Path(match).resolve()
            if path not in seen:
                seen.add(path)
                paths.append(path)
    return paths


def jsonl(path):
    with path.open() as stream:
        for line, text in enumerate(stream, 1):
            if not text.strip():
                continue
            try:
                yield json.loads(text), {"path": str(path), "line": line}
            except json.JSONDecodeError as exc:
                raise ValueError(f"Known repeats must be finished: invalid JSONL at {path}:{line}") from exc


def current_id(shape, row, metadata, common, *, location, allow_unavailable=False):
    raw_id = row.get("kid", row.get("kernelId"))
    name = row.get("kernelName")
    try:
        kid = int(raw_id)
    except (ValueError, TypeError):
        raise ValueError(f"Missing current numeric candidate ID at {location}: {raw_id}")
    if kid not in metadata:
        if allow_unavailable:
            return None
        raise ValueError(f"Unavailable current kid {kid} at {location}")
    if name != metadata[kid]["kernelName"]:
        if allow_unavailable:
            return None
        raise ValueError(f"Current complete name/ID mismatch at {location}: kid={kid}")
    for field in ("traits", "geometry"):
        if field in row and row[field] != metadata[kid][field]:
            if allow_unavailable:
                return None
            raise ValueError(f"Current {field} differs from registry at {location}: kid={kid}")
    if id(common) not in _REGISTRY_CACHE:
        _REGISTRY_CACHE[id(common)] = common.get_registry("a8w8_blockscale_bpreshuffle", "gfx950", 256)
    reg = _REGISTRY_CACHE[id(common)]
    if not common.a8w8_mxscale_bpreshuffle_supports_shape(reg[kid], *shape[2:5]):
        if allow_unavailable:
            return None
        raise ValueError(f"Current kid {kid} is illegal for shape {shape} at {location}")
    return kid


def known_best(paths, expected, baseline, metadata, common, limit):
    groups, excluded, counts = defaultdict(dict), [], Counter()
    for path in paths:
        batch = repeat_batch(path)
        for row, source in jsonl(path):
            shape = strict_key(row["shape"], source)
            if shape not in expected:
                continue
            counts["records_read"] += 1
            kid = current_id(shape, row, metadata, common, location=source, allow_unavailable=True)
            if kid is None:
                counts["unavailable_or_shape_illegal_records"] += 1
                raw_kid = int(row.get("kid", row.get("kernelId")))
                reason = ("unavailable_current_kid" if raw_kid not in metadata else
                          "current_name_mismatch" if row.get("kernelName") != metadata[raw_kid]["kernelName"] else
                          "outside_production_shape_or_config")
                counts[reason + "_records"] += 1
                continue
            if type(row.get("round")) is not int or row["round"] not in ROUNDS:
                raise ValueError(f"Unexpected known-repeat round at {source}")
            if number(row.get("csv_baseline_us")) != baseline[shape]:
                raise ValueError(f"Known repeat baseline differs from fixed CSV at {source}")
            identity = (shape, kid, batch)
            index = row["round"]
            if index in groups[identity]:
                raise ValueError(f"Duplicate known-repeat shape/kid/round within batch {batch} at {source}")
            groups[identity][index] = {"record": row, "source": source}
    latest = {}
    for (shape, kid, batch), rounds in groups.items():
        complete = set(rounds) == ROUNDS
        entries = [rounds[index] for index in sorted(rounds)]
        records = [entry["record"] for entry in entries]
        files = {entry["source"]["path"] for entry in entries}
        gpus = {record.get("gpu_logical_id") for record in records}
        reasons = []
        if not complete:
            reasons.append("incomplete_six_rounds")
        if not all(valid(row, limit) for row in records):
            reasons.append("incorrect_or_failed_round")
        if not all(type(row.get("rotate_args")) is int and row["rotate_args"] == 0 for row in records):
            reasons.append("hot_or_unproven_cache_rotation")
        if len(files) != 1 or len(gpus) != 1 or not all(type(gpu) is int and gpu in range(4) for gpu in gpus):
            reasons.append("split_or_missing_file_gpu_identity")
        timestamp = max(row.get("completed_utc", "") for row in records)
        if reasons:
            excluded.append({**shape_row(shape), "kid": kid, "batch": batch, "reasons": reasons})
        # An incomplete batch cannot displace a prior complete batch. A newer
        # complete failed/hot batch makes the old candidate unverified.
        if complete and timestamp >= latest.get((shape, kid), {}).get("timestamp", ""):
            latest[(shape, kid)] = {"valid": not reasons, "timestamp": timestamp,
                                   "kid": kid, "kernelName": metadata[kid]["kernelName"],
                                   "us": statistics.median(float(row["us"]) for row in records) if not reasons else None,
                                   "errRatio": max(float(row["errRatio"]) for row in records) if not reasons else None,
                                   "batch": batch, "sources": [entry["source"] for entry in entries]}
    best = {}
    for (shape, kid), result in latest.items():
        if result["valid"] and (shape not in best or (result["us"], kid) < (best[shape]["us"], best[shape]["kid"])):
            best[shape] = result
    counts.update(candidate_batches=len(groups), selected_best_shapes=len(best), excluded_batches=len(excluded))
    return best, {"counts": dict(counts), "excluded_batches": excluded}


def current_verified(paths, expected, baseline, metadata, common, limit):
    additions, skipped = defaultdict(list), []
    def add(shape, row, source, role, *, allow_unknown_error=False):
        if shape not in expected:
            return
        us, err = number(row.get("us")), number(row.get("errRatio"))
        if us is None or us <= 0 or (not allow_unknown_error and (err is None or not 0 <= err <= limit)):
            skipped.append({"source": source, "reason": "invalid verified timing/correctness fields"})
            return
        kid = current_id(shape, row, metadata, common, location=source, allow_unavailable=True)
        if kid is None:
            skipped.append({"source": source, "reason": "verified candidate not available/legal in current public registry"})
            return
        additions[shape].append({"kid": kid, "kernelName": metadata[kid]["kernelName"],
                                 "us": us, "errRatio": err, "role": role, "source": source})
    for path in paths:
        report = json.loads(path.read_text())
        candidates = {(strict_key(row, f"{path}:candidates"), int(row["kid"])): row
                      for row in report.get("candidates", []) if "kid" in row}
        for index, row in enumerate(report.get("shapes", [])):
            shape = strict_key(row, f"{path}:shapes[{index}]")
            source = {"path": str(path), "section": "shapes", "index": index}
            if row.get("evidence_kind", row.get("kind")) == "repeat_median":
                if row.get("status") not in ("no_correct_opus", "pending") and boolean(row.get("correct", True)):
                    add(shape, row, source, "current_verified_best")
            elif boolean(row.get("repeat_complete")) and row.get("best_repeat_kid") is not None:
                kid = int(row["best_repeat_kid"])
                result = candidates.get((shape, kid))
                if result and all(boolean(result.get(field)) for field in
                                  ("rounds_complete", "all_rounds_correct", "timing_matches_tuner")):
                    if number(result.get("median_us")) != number(row.get("best_repeat_median_us")):
                        raise ValueError(f"Current report selected median disagrees with candidate at {source}")
                    add(shape, {**result, "us": result["median_us"], "errRatio": result.get("max_errRatio")},
                        source, "current_verified_best")
        # An integration receipt has already recomputed raw six-round pair
        # statistics. Preserve its stable new COs even if another pipeline was
        # faster overall. Never claim a missing max error as zero.
        selected_ids = set(map(int, report.get("selected_new_ids", [])))
        for index, row in enumerate(report.get("parent_comparisons", [])):
            stable = row.get("selected_stable_improvement", row.get("stable_improvement"))
            if not boolean(stable) or selected_ids and int(row["new_kid"]) not in selected_ids:
                continue
            shape = strict_key(row, f"{path}:parent_comparisons[{index}]")
            if shape not in expected:
                continue
            if number(row.get("csv_baseline_us")) != baseline[shape]:
                raise ValueError("Integration receipt comparison baseline differs from fixed CSV")
            add(shape, {"kid": row["new_kid"], "kernelName": row["kernelName"],
                        "us": row.get("current_median_us", row.get("new_median_us")),
                        "errRatio": row.get("max_errRatio")},
                {"path": str(path), "section": "parent_comparisons", "index": index},
                "current_verified_new_co", allow_unknown_error=True)
    return additions, skipped


def known_csv_candidates(paths, expected, metadata, common):
    """Shape-specific full identities generated by focused selection helpers."""
    entries, outside = defaultdict(dict), 0
    for path in paths:
        for line, row in enumerate(csv_rows(path), 2):
            # This helper schema omits constant namespace fields by design.
            shape = strict_key({"gfx": "gfx950", "cu_num": 256,
                                "w_scale_block": "128x128", **row}, f"{path}:{line}")
            if shape not in expected:
                outside += 1
                continue
            normalized = dict(row)
            if isinstance(normalized.get("geometry"), str):
                normalized["geometry"] = json.loads(normalized["geometry"])
            if normalized.get("kind") == "CO" and isinstance(normalized.get("traits"), str):
                normalized["traits"] = json.dumps(json.loads(normalized["traits"]), sort_keys=True,
                                                 separators=(",", ":"))
            if any(field not in normalized or normalized[field] in (None, "")
                   for field in ("kid", "kernelName", "kind", "geometry", "traits")):
                raise ValueError(f"Known CSV lacks complete current identity at {path}:{line}")
            kid = current_id(shape, normalized, metadata, common, location=f"{path}:{line}")
            if normalized["kind"] != metadata[kid]["kind"]:
                raise ValueError(f"Known CSV kind differs from current registry at {path}:{line}")
            roles = normalized.get("role", "known_winner")
            if isinstance(roles, str):
                roles = [value.strip() for value in roles.split("|") if value.strip()]
            existing = entries[shape].setdefault(kid, {"roles": set(), "sources": []})
            existing["roles"].update(roles or ["known_winner"])
            existing["sources"].append({"path": str(path), "line": line})
    return entries, outside


def prepare(args):
    shape_rows = csv_rows(args.shapes)
    shapes = sorted({strict_key(row, args.shapes) for row in shape_rows})
    if not shapes or len(shapes) != len(shape_rows):
        raise ValueError("Expected shapes must be nonempty and have unique six-field keys")
    expected = set(shapes)
    baseline = {}
    for row in csv_rows(args.baseline):
        shape = strict_key(row, args.baseline, require_scope=False)
        if shape not in expected:
            continue
        us = number(row.get("us"))
        if shape in baseline or us is None or us <= 0:
            raise ValueError(f"Duplicate/invalid fixed baseline for {shape}")
        baseline[shape] = us
    if set(baseline) != expected:
        raise ValueError("Fixed baseline does not cover every expected shape")
    metadata, pairs = registry_metadata()
    common = pure_common()
    known_kids = set(map(int, getattr(args, "known_kids", [])))
    if known_kids - metadata.keys():
        raise ValueError(f"Explicit --known-kids are unavailable in the current production registry: {sorted(known_kids - metadata.keys())}")
    indexes = name_indexes(metadata)
    profile_paths, repeat_paths, report_paths = expand(args.profile), expand(args.known_repeat), expand(args.current_report)
    known_csv_paths = expand(getattr(args, "known_csv", []))
    inputs = list(dict.fromkeys([args.shapes.resolve(), args.baseline.resolve(), args.history.resolve(),
                                *profile_paths, *repeat_paths, *report_paths, *known_csv_paths]))
    source_inputs = [ROOT / "csrc/opus_gemm/opus_gemm_common.py", ROOT / "csrc/opus_gemm/gen_co/co_kernels_gfx950.json"]
    input_hashes = {str(path): file_hash(path) for path in inputs + source_inputs}
    observed = defaultdict(dict)
    profile_status = defaultdict(dict)
    for path in profile_paths:
        for line, row in enumerate(csv_rows(path), 2):
            if row.get("libtype") != "opus":
                continue
            shape = strict_key(row, f"{path}:{line}")
            if shape not in expected:
                continue
            if not valid(row, args.err_ratio):
                # The latest failed observation of a numeric current kid
                # supersedes its earlier correct row. Normal invalid profile
                # names are "None"; they never become selected identities.
                try:
                    failed_kid = int(row.get("kernelId"))
                except (TypeError, ValueError):
                    failed_kid = None
                if failed_kid in metadata:
                    observed[shape].pop(failed_kid, None)
                    profile_status[shape][failed_kid] = {"status": "incorrect_profile_observation",
                                                        "us": number(row.get("us")), "errRatio": number(row.get("errRatio")),
                                                        "source": {"path": str(path), "line": line}}
                continue
            kid = current_id(shape, row, metadata, common, location=f"{path}:{line}")
            observed[shape][kid] = {"kid": kid, "kernelName": metadata[kid]["kernelName"],
                                    "us": float(row["us"]), "errRatio": float(row["errRatio"]),
                                    "source": {"path": str(path), "line": line}}
            profile_status[shape][kid] = {"status": "correct", **observed[shape][kid]}
    absent = [shape for shape in shapes if not observed[shape]]
    if absent:
        raise ValueError(f"Final profiles have no correct OPUS candidate for {len(absent)}/{len(shapes)} shapes; finish full421/fallback first")
    history = {}
    for row in csv_rows(args.history):
        shape = strict_key(row, args.history)
        if shape in expected:
            if shape in history:
                raise ValueError("Duplicate remote history shape key")
            history[shape] = row
    if set(history) != expected:
        raise ValueError("Remote full-name history does not cover all expected shapes")
    known, known_receipt = known_best(repeat_paths, expected, baseline, metadata, common, args.err_ratio)
    verified, skipped_current = current_verified(report_paths, expected, baseline, metadata, common, args.err_ratio)
    known_csv, outside_known_csv = known_csv_candidates(known_csv_paths, expected, metadata, common)
    plan_shapes, shape_receipts = [], []
    roles_count, selected_pipelines = Counter(), Counter()
    known_kid_counts = Counter()
    known_csv_omissions, known_csv_selected = [], 0
    for shape in shapes:
        valid_results = sorted(observed[shape].values(), key=lambda item: (item["us"], item["kid"]))
        by_pipeline = defaultdict(list)
        for result in valid_results:
            by_pipeline[metadata[result["kid"]]["geometry"][0]].append(result)
        mapping = map_remote(history[shape]["opus_kernel"], metadata, indexes, shape)
        roles, evidence = defaultdict(set), defaultdict(list)
        for result in valid_results[:args.topk]:
            roles[result["kid"]].add("global_top5" if args.topk == 5 else "global_topk")
        for pipeline, rows in by_pipeline.items():
            for result in rows[:args.pipeline_topk]:
                roles[result["kid"]].add("pipeline_top2" if args.pipeline_topk == 2 else "pipeline_topk")
        for kid in mapping["legal_current_kids"]:
            roles[kid].add("remote_exact")
        for kid in known_kids:
            if kid in observed[shape]:
                roles[kid].add("known_production_kid")
                known_kid_counts[kid] += 1
        for kid, known_item in known_csv.get(shape, {}).items():
            if kid not in observed[shape]:
                failure = profile_status[shape].get(kid, {"status": "missing_profile_observation"})
                known_csv_omissions.append({**shape_row(shape), "kid": kid,
                                           "kernelName": metadata[kid]["kernelName"],
                                           "role": "|".join(sorted(known_item["roles"])),
                                           **failure, "known_sources": known_item["sources"]})
                continue
            roles[kid].add("known_csv")
            roles[kid].update("known_csv:" + role for role in known_item["roles"])
            known_csv_selected += 1
        if shape in known:
            kid = known[shape]["kid"]
            roles[kid].add("known_six_round_best")
            evidence[kid].append(known[shape])
        for item in verified.get(shape, []):
            roles[item["kid"]].add(item["role"])
            evidence[item["kid"]].append(item)
        ordered = sorted(roles, key=lambda kid: (observed[shape].get(kid, {}).get("us", math.inf), kid))
        candidates = []
        for kid in ordered:
            profile = observed[shape].get(kid)
            result = {"us": profile["us"] if profile else None,
                      "errRatio": profile["errRatio"] if profile else None,
                      "correct": profile is not None, "status": "correct" if profile else "not_in_final_profile"}
            candidate = {**metadata[kid], **result, "roles": sorted(roles[kid]),
                         "exact_opposite_kids": pairs[kid], "verified_evidence": evidence.get(kid, []),
                         "initial_source": profile["source"] if profile else None,
                         "remote_reference_us": number(history[shape]["opus_us"]) if "remote_exact" in roles[kid] else None}
            candidates.append(candidate)
            roles_count.update(roles[kid])
            selected_pipelines[metadata[kid]["geometry"][0]] += 1
        plan_shapes.append({"shape": shape_row(shape), "csv_baseline_us": baseline[shape],
                            "remote_winner": {"kernelName": history[shape]["opus_kernel"],
                                              "us": number(history[shape]["opus_us"]),
                                              "is_co": boolean(history[shape].get("best_opus_is_co")),
                                              "mapping": mapping}, "candidates": candidates})
        shape_receipts.append({**shape_row(shape), "correct_profile_candidates": len(valid_results),
                               "selected_candidates": len(candidates), "remote_mapped": mapping["mapped"],
                               "remote_mapping_method": mapping["method"],
                               "known_repeat_best_kid": known[shape]["kid"] if shape in known else None,
                               "verified_current_candidates": len(verified.get(shape, [])),
                               "known_csv_required_candidates": len(known_csv.get(shape, {})),
                               "known_csv_correct_candidates_selected": sum(kid in observed[shape] for kid in known_csv.get(shape, {})),
                               "selected_4wave_candidates": sum(candidate["geometry"][0] == "pin_agpr" for candidate in candidates)})
    if input_hashes != {path: file_hash(Path(path)) for path in input_hashes}:
        raise ValueError("Profile/repeat/report/core inputs changed during preparation; retry against finished files")
    plan = {"version": 2, "family": "a8w8_blockscale_bpreshuffle", "metadata_module": "prepare_opus_repeat",
            "source_profiles": list(map(str, profile_paths)), "correctness_limit": args.err_ratio,
            "selection": f"Correct global Top{args.topk}, per-pipeline Top{args.pipeline_topk}, exact remote names, known complete six-round best and current verified best/new COs",
            "shapes": plan_shapes}
    receipt = {"generated_utc": datetime.now(timezone.utc).isoformat(), "expected_shapes": len(shapes),
               "profile_complete_shapes": len(shapes)-len(absent), "repeat_shapes": len(plan_shapes),
               "repeat_candidates": sum(len(item["candidates"]) for item in plan_shapes),
               "unique_kids": len({candidate["kid"] for item in plan_shapes for candidate in item["candidates"]}),
               "roles": dict(roles_count), "by_pipeline": dict(selected_pipelines),
               "remote_mapped_shapes": sum(row["remote_mapped"] for row in shape_receipts),
               "known_repeat_best_shapes": len(known),
               "current_verified_shape_count": len(verified), "known_repeat_audit": known_receipt,
               "explicit_known_kids": sorted(known_kids),
               "explicit_known_kid_correct_profile_shapes": {str(kid): known_kid_counts[kid] for kid in sorted(known_kids)},
               "known_csv_required_candidates": sum(len(items) for items in known_csv.values()),
               "known_csv_candidates_selected": known_csv_selected,
               "known_csv_omission_count": len(known_csv_omissions), "known_csv_omissions": known_csv_omissions,
               "known_csv_outside_expected_rows": outside_known_csv,
               "skipped_current_report_records": skipped_current,
               "inputs": input_hashes, "torch_imported": "torch" in sys.modules,
               "aiter_imported": "aiter" in sys.modules, "gpu_used": False, "jit_or_co_built": False}
    return plan, receipt, shape_receipts


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", nargs="+", required=True, help="Completed full421/fallback profile CSV paths or quoted globs")
    parser.add_argument("--shapes", type=Path, default=DOC / "shapes_421.csv")
    parser.add_argument("--baseline", type=Path, default=DATA / "csv_baseline.csv")
    parser.add_argument("--history", type=Path, default=DOC / "all_shapes_421.csv")
    parser.add_argument("--known-repeat", nargs="*", default=[], help="Original32/46 and stage3 completed six-round raw JSONL paths or quoted globs")
    parser.add_argument("--known-kids", type=int, nargs="*", default=[], help="Retain these production IDs for every expected shape with a correct profile observation")
    parser.add_argument("--known-csv", nargs="*", default=[], help="Shape-specific known_winners.csv with M,N,K,kid,kernelName,kind,role,geometry,traits")
    parser.add_argument("--current-report", nargs="*", default=[], help="finish_workflow/summarize_existing_repeat reports or integration receipts containing verified best/new COs")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--pipeline-topk", type=int, default=2)
    parser.add_argument("--err-ratio", type=float, default=.05)
    parser.add_argument("--output-dir", type=Path, required=True, help="Fresh plan/receipt directory")
    args = parser.parse_args()
    if args.topk < 1 or args.pipeline_topk < 1 or not 0 <= args.err_ratio <= .05 or not math.isfinite(args.err_ratio):
        parser.error("Use positive TopK values and finite err-ratio in [0,0.05]")
    output = args.output_dir.resolve()
    if output.exists():
        parser.error("Choose a fresh output directory to preserve prepared/final repeat plans")
    try:
        plan, receipt, shape_receipts = prepare(args)
        if receipt["torch_imported"] or receipt["aiter_imported"]:
            raise ValueError("CPU preparation unexpectedly imported GPU/JIT modules")
        output.mkdir(parents=True)
        plan_path = output / "repeat_plan.json"
        plan_path.write_text(json.dumps(plan, indent=2, allow_nan=False) + "\n")
        (output / "shape_receipt.json").write_text(json.dumps(shape_receipts, indent=2, allow_nan=False) + "\n")
        if receipt["known_csv_omissions"]:
            from audit_progress import write_csv
            omissions = [{field: json.dumps(value) if isinstance(value, (dict, list)) else value
                          for field, value in row.items()} for row in receipt["known_csv_omissions"]]
            write_csv(output / "known_csv_omissions.csv", omissions,
                      ("gfx", "cu_num", "M", "N", "K", "w_scale_block", "kid", "kernelName", "role", "status", "us", "errRatio", "source", "known_sources"))
        runner = LOCAL / "repeat_opus_candidates.py"
        dry = subprocess.run([sys.executable, str(runner), "--plan", str(plan_path), "--rounds", "6", "--iters", "101", "--dry-run"],
                             capture_output=True, text=True, check=True)
        dry_report = json.loads(dry.stdout)
        if dry_report["shapes"] != len(plan["shapes"]) or dry_report["torch_imported"]:
            raise ValueError("Actual runner CPU dry-run did not preserve complete coverage")
        receipt.update(cpu_dry_run_passed=True, cpu_dry_run=dry_report)
        (output / "receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
        print(json.dumps({field: receipt[field] for field in (
            "expected_shapes", "profile_complete_shapes", "repeat_shapes", "repeat_candidates", "unique_kids",
            "remote_mapped_shapes", "known_repeat_best_shapes", "current_verified_shape_count",
            "known_csv_required_candidates", "known_csv_candidates_selected", "known_csv_omission_count",
            "cpu_dry_run_passed", "torch_imported", "gpu_used", "jit_or_co_built")}, indent=2))
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as exc:
        if isinstance(exc, subprocess.CalledProcessError):
            parser.error(f"Runner CPU dry-run failed: {exc.stderr}")
        parser.error(str(exc))


if __name__ == "__main__":
    main()
