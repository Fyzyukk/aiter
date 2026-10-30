#!/usr/bin/env python3
"""CPU-only NT accounting across directed runs, full Top8 and fallback runs.

All OPUS rows in --profile participate in candidate discovery. A complete,
correct six-round repeat replaces every single-pass observation of the same
six-field shape key and complete kernel name. A complete repeat with a failed
round invalidates that candidate. Different repeat batches never mix rounds.

Example (quote globs so this program expands them):
  python3 finish_workflow.py --profile step1_profile.csv step2_profile.csv \
    'step3_*/profile.csv' full421/profile.csv --progress step1_progress.jsonl \
    step2_progress.jsonl 'step3_*/progress.jsonl' --full-progress full421/progress.jsonl \
    --repeat 'step1_repeat_shard*.jsonl' 'step2_repeat_shard*.jsonl' \
    'step3_*/repeat_shard*.jsonl' 'full421/repeat_shard*.jsonl'

Historical handoff records are never counted as current wins. This script
imports only standard-library modules and the CPU audit helpers.
"""

import argparse
import csv
import glob
import json
import math
import os
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from audit_progress import DATA, DOC, SHAPE_FIELDS, csv_rows, m_class, name_geometry

LOCAL = Path(__file__).resolve().parent
CONFIG_FIELDS = SHAPE_FIELDS + ("libtype", "kernelId", "splitK", "us", "kernelName", "errRatio")
RESULT_FIELDS = CONFIG_FIELDS + ("status", "csv_baseline_us", "wins_csv", "speedup", "evidence_kind",
                                "wins_csv_every_round", "confirmation_pending", "complete_repeat_batches",
                                "pipeline", "candidate_count", "correct_candidate_count", "best_sources",
                                "discovery_best", "discovery_single_pass_best",
                                "discovery_single_pass_faster_than_repeat", "correct_repeat_candidate_count")


def finite(value):
    try:
        value = float(value)
    except (ValueError, TypeError):
        return None
    return value if math.isfinite(value) else None


def boolean(value):
    return value is True or str(value).lower() == "true"


def strict_key(row, location, require_scope=True):
    absent = [field for field in SHAPE_FIELDS if field not in row or row[field] is None]
    if absent:
        raise ValueError(f"Missing shape fields {absent} at {location}")
    shape = (str(row["gfx"]), int(row["cu_num"]), int(row["M"]), int(row["N"]),
             int(row["K"]), str(row["w_scale_block"]))
    if min(shape[2:5]) <= 0 or require_scope and (shape[0] != "gfx950" or shape[1] != 256 or shape[5] != "128x128"):
        raise ValueError(f"Unexpected gfx950 NT shape key {shape} at {location}")
    return shape


def shape_row(shape):
    return dict(zip(SHAPE_FIELDS, shape))


def paths(patterns, missing):
    """Preserve caller order and report absent paths without breaking live reports."""
    found, seen = [], set()
    for pattern in patterns or []:
        matches = sorted(glob.glob(str(pattern)))
        if not matches:
            missing.append(str(pattern))
        for match in matches:
            path = Path(match).resolve()
            if path not in seen:
                seen.add(path)
                found.append(path)
    return found


def jsonl(path, partial):
    with path.open() as stream:
        for line, text in enumerate(stream, 1):
            if not text.strip():
                continue
            try:
                record = json.loads(text)
            except json.JSONDecodeError as exc:
                # Only an unfinished final line is acceptable for a live writer.
                if not text.endswith("\n"):
                    partial.append(f"{path}:{line}")
                    continue
                raise ValueError(f"Invalid JSONL at {path}:{line}: {exc}") from exc
            yield record, {"path": str(path), "line": line,
                           "completed_utc": record.get("completed_utc", "")}


def valid(record, limit):
    us, err = finite(record.get("us")), finite(record.get("errRatio"))
    explicit_correct = record.get("correct", True)
    return bool(us is not None and us > 0 and err is not None and 0 <= err <= limit
                and boolean(explicit_correct))


def candidate(record, source, limit):
    kid = record.get("kernelId", record.get("kid"))
    if kid is not None and str(kid) not in ("", "None"):
        kid = int(kid)
    return {"kernelId": kid, "splitK": int(record.get("splitK", 1)),
            "kernelName": record["kernelName"], "us": finite(record.get("us")),
            "errRatio": finite(record.get("errRatio")), "correct": valid(record, limit),
            "kind": "single_pass", "sources": [source], "repeat": None}


def repeat_batch(path):
    # The GPU shard is not an independent repeat batch. The surrounding run
    # directory and the remaining stem distinguish step1, step2, stage3, etc.
    stem = re.sub(r"(?:_?shard|_?gpu)\d+(?=_|$)", "", path.stem)
    return str(path.parent / stem)


def file_time(path):
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()


def sort_time(source):
    return source.get("completed_utc") or source.get("file_completed_utc", "")


def summarize_group(rows):
    counts = Counter(row["status"] for row in rows)
    return {"expected": len(rows), "completed": len(rows) - counts["pending"],
            "wins_csv": counts["wins_csv"], "losses_csv": counts["slower_than_csv"],
            "no_correct_opus": counts["no_correct_opus"], "pending": counts["pending"],
            "repeat_median_best": sum(row["evidence_kind"] == "repeat_median" for row in rows),
            "single_pass_best": sum(row["evidence_kind"] == "single_pass" for row in rows),
            "confirmation_pending": sum(row["confirmation_pending"] for row in rows)}


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


def atomic_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                             for field, value in row.items()})
    os.replace(temporary, path)


def analyze(args):
    expected_rows = csv_rows(args.expected)
    expected = {strict_key(row, args.expected): row for row in expected_rows}
    if len(expected) != len(expected_rows):
        raise ValueError(f"Duplicate expected shape keys in {args.expected}")
    baseline, baseline_names = {}, {}
    for line, row in enumerate(csv_rows(args.baseline), 2):
        shape = strict_key(row, f"{args.baseline}:{line}", require_scope=False)
        if shape not in expected:
            continue  # The saved CSV also contains other architectures/scale layouts.
        us = finite(row.get("us"))
        if shape in baseline or us is None or us <= 0:
            raise ValueError(f"Duplicate key or invalid baseline at {args.baseline}:{line}")
        baseline[shape], baseline_names[shape] = us, row.get("kernelName")
    if set(expected) - baseline.keys():
        raise ValueError(f"Missing fixed CSV baselines for {len(set(expected) - baseline.keys())} expected shapes")
    missing, partial = [], []
    progress_paths = paths(args.progress, missing)
    full_paths = paths(args.full_progress, missing)
    fallback_paths = paths(args.fallback_progress, missing)
    profile_paths = paths(args.profile, missing)
    repeat_paths = paths(args.repeat, missing)
    fallback_expected = ({strict_key(row, args.fallback_expected) for row in csv_rows(args.fallback_expected)}
                         if args.fallback_expected else set())
    if fallback_expected - expected.keys():
        raise ValueError("Fallback shape list contains shapes outside the expected full list")
    counts, observed, full_done, fallback_done = Counter(), set(), set(), set()
    use_current_registry = bool(getattr(args, "current_registry", False))
    filter_to_expected = bool(getattr(args, "filter_to_expected", False))
    require_repeat_winners = bool(getattr(args, "require_repeat_winners", False))
    production_common, production_registry = None, None
    excluded_candidates = {}
    if use_current_registry:
        from prepare_opus_repeat import pure_common
        production_common = pure_common()
        production_registry = production_common.get_registry("a8w8_blockscale_bpreshuffle", "gfx950", 256)
    single, single_sequence = defaultdict(dict), {}
    repeated = defaultdict(dict)
    # Names keep identity stable across sparse CO additions and frozen JIT IDs.
    name_kids = {}
    def identity(shape, record, source):
        name = record.get("kernelName")
        if not name or name == "None":
            return None
        if not str(name).startswith("opus_gemm_"):
            raise ValueError(f"Unexpected OPUS kernel name at {source}: {name}")
        if re.search(r"(?:^|_)(?:tt|nn|tn)(?:\d|_|$)", name):
            raise ValueError(f"Non-NT candidate at {source}: {name}")
        kid = record.get("kernelId", record.get("kid"))
        if production_registry is not None:
            numeric_kid = int(kid) if kid is not None and str(kid) not in ("", "None") else None
            reason = ("missing_current_kid" if numeric_kid is None else
                      "unavailable_current_kid" if numeric_kid not in production_registry else
                      "current_name_mismatch" if name != production_registry[numeric_kid].name else
                      "outside_production_supported_shapes" if not production_common.a8w8_mxscale_bpreshuffle_supports_shape(
                          production_registry[numeric_kid], *shape[2:5]) else None)
            if reason:
                identity_key = (shape, numeric_kid, name, reason)
                if identity_key not in excluded_candidates:
                    excluded_candidates[identity_key] = {**shape_row(shape), "kernelId": numeric_kid,
                                                          "kernelName": name, "reason": reason,
                                                          "observations": 0, "sources": []}
                excluded_candidates[identity_key]["observations"] += 1
                excluded_candidates[identity_key]["sources"].append(source)
                counts["excluded_experiment_observations"] += 1
                return None
        if kid is not None and str(kid) not in ("", "None"):
            kid = int(kid)
            old = name_kids.setdefault((shape, name), kid)
            if old != kid:
                raise ValueError(f"Same complete name has conflicting current IDs at {source}: {old}/{kid}")
        return name
    def measure_count(record, prefix):
        counts[prefix + "_measurements"] += 1
        err = finite(record.get("errRatio"))
        if err is None or err < 0 or err > args.err_ratio or not boolean(record.get("correct", True)):
            counts[prefix + "_incorrect_measurements"] += 1
        us = finite(record.get("us"))
        if us is None or us <= 0:
            counts[prefix + "_invalid_time_measurements"] += 1
    def put_single(shape, record, source, sequence):
        name = identity(shape, record, source)
        if name is None:
            return
        item = candidate(record, source, args.err_ratio)
        ordinal = (sort_time(source), sequence)
        previous = single_sequence.get((shape, name))
        if previous is None or ordinal >= previous:
            single[shape][name] = item
            single_sequence[(shape, name)] = ordinal
    # Reading profiles first allows progress-only runs while preserving all
    # secondary candidates in complete profiles. A progress winner from that
    # same run is merely a duplicate discovery observation.
    sequence = 0
    for path in profile_paths:
        timestamp = file_time(path)
        with path.open(newline="", encoding="utf-8-sig") as stream:
            for line, row in enumerate(csv.DictReader(stream), 2):
                if row.get("libtype") != "opus":
                    continue
                shape = strict_key(row, f"{path}:{line}")
                if shape not in expected:
                    if filter_to_expected:
                        counts["outside_expected_profile_observations"] += 1
                        continue
                    raise ValueError(f"Unexpected profile shape at {path}:{line}: {shape}")
                observed.add(shape)
                measure_count(row, "single_pass")
                sequence += 1
                put_single(shape, row, {"path": str(path), "line": line,
                                       "file_completed_utc": timestamp}, sequence)
    for path in dict.fromkeys(progress_paths + full_paths + fallback_paths):
        latest = {}
        for record, source in jsonl(path, partial):
            shape = strict_key(record["shape"], source)
            if shape not in expected:
                if filter_to_expected:
                    counts["outside_expected_progress_observations"] += 1
                    continue
                raise ValueError(f"Unexpected progress shape at {source}: {shape}")
            latest[shape] = (record, source)
        for shape, (record, source) in latest.items():
            observed.add(shape)
            if path in full_paths:
                full_done.add(shape)
            if path in fallback_paths:
                fallback_done.add(shape)
            result = record.get("best_opus")
            if result:
                sequence += 1
                counts["progress_best_observations"] += 1
                put_single(shape, result, source, sequence)
    for path in repeat_paths:
        batch = repeat_batch(path)
        for record, source in jsonl(path, partial):
            shape = strict_key(record["shape"], source)
            if shape not in expected:
                if filter_to_expected:
                    counts["outside_expected_repeat_observations"] += 1
                    continue
                raise ValueError(f"Unexpected repeat shape at {source}: {shape}")
            measure_count(record, "repeat")
            if int(record.get("rotate_args", 0)) != 0:
                counts["excluded_hot_cache_repeat_measurements"] += 1
                continue
            observed.add(shape)
            name = identity(shape, record, source)
            if name is None:
                if use_current_registry and record.get("kernelName") not in (None, "", "None"):
                    continue
                raise ValueError(f"Missing complete name in repeat at {source}")
            round_id = int(record["round"])
            if round_id not in range(args.rounds):
                raise ValueError(f"Unexpected repeat round {round_id} at {source}; expected 0..{args.rounds - 1}")
            group = repeated[(shape, name, batch)]
            if round_id in group:
                raise ValueError(f"Duplicate shape/name/round within batch {batch}: {source}")
            if "csv_baseline_us" in record and finite(record["csv_baseline_us"]) != baseline[shape]:
                raise ValueError(f"Repeat saved baseline differs from fixed CSV at {source}")
            group[round_id] = {"record": record, "source": source}
    final = defaultdict(dict, {shape: dict(items) for shape, items in single.items()})
    repeat_details, pending_names, complete_names = [], set(), Counter()
    latest_complete = {}
    for (shape, name, batch), rounds in repeated.items():
        complete = set(rounds) == set(range(args.rounds))
        entries = [rounds[index] for index in sorted(rounds)]
        source_files = {entry["source"]["path"] for entry in entries}
        gpu_ids = {entry["record"].get("gpu_logical_id") for entry in entries}
        batch_integrity = (len(source_files) == 1 and len(gpu_ids) == 1
                           and all(type(gpu) is int and gpu in range(4) for gpu in gpu_ids))
        correct = (complete and all(valid(entry["record"], args.err_ratio) for entry in entries)
                   and (not require_repeat_winners or batch_integrity))
        times = [finite(entry["record"].get("us")) for entry in entries]
        valid_times = [time for time in times if time is not None and time > 0]
        detail = {**shape_row(shape), "kernelName": name, "batch": batch, "rounds_seen": sorted(rounds),
                  "complete": complete, "all_rounds_correct": correct,
                  "batch_integrity_verified": batch_integrity,
                  "median_us": statistics.median(valid_times) if valid_times else None,
                  "min_us": min(valid_times) if valid_times else None,
                  "max_us": max(valid_times) if valid_times else None,
                  "max_errRatio": max((finite(entry["record"].get("errRatio")) or 0 for entry in entries), default=None)}
        repeat_details.append(detail)
        counts["repeat_candidate_batches"] += 1
        if not complete:
            counts["incomplete_repeat_candidate_batches"] += 1
            pending_names.add((shape, name))
            continue
        counts["complete_repeat_candidate_batches"] += 1
        counts["correct_complete_repeat_candidate_batches"] += int(correct)
        complete_names[(shape, name)] += 1
        timestamp = max(sort_time(entry["source"]) for entry in entries)
        if timestamp < latest_complete.get((shape, name), ""):
            continue
        latest_complete[(shape, name)] = timestamp
        record = entries[0]["record"]
        final[shape][name] = {"kernelId": int(record.get("kid", record.get("kernelId"))), "splitK": 1,
                              "kernelName": name, "us": detail["median_us"],
                              "errRatio": detail["max_errRatio"], "correct": correct,
                              "kind": "repeat_median", "sources": [entry["source"] for entry in entries],
                              "repeat": detail}
    rows = []
    for shape in sorted(expected):
        correct_candidates = [record for record in final[shape].values() if record["correct"]]
        repeated_candidates = [record for record in correct_candidates if record["kind"] == "repeat_median"]
        single_candidates = [record for record in correct_candidates if record["kind"] == "single_pass"]
        latency_order = lambda record: (record["us"], record["kernelName"])
        discovery = min(correct_candidates, key=latency_order) if correct_candidates else None
        discovery_single = min(single_candidates, key=latency_order) if single_candidates else None
        repeat_best = min(repeated_candidates, key=latency_order) if repeated_candidates else None
        available = repeated_candidates if require_repeat_winners else correct_candidates
        best = min(available, key=lambda record: (record["us"], record["kernelName"])) if available else None
        wins = bool(best and best["us"] < baseline[shape])
        geometry = name_geometry(best["kernelName"]) if best else None
        confirmation_pending = bool(best and (best["kind"] == "single_pass" or (shape, best["kernelName"]) in pending_names))
        row = {**shape_row(shape), "libtype": "opus", "kernelId": best["kernelId"] if best else None,
               "splitK": best["splitK"] if best else None, "us": best["us"] if best else None,
               "kernelName": best["kernelName"] if best else None, "errRatio": best["errRatio"] if best else None,
               "status": "pending" if shape not in observed else "no_correct_opus" if not best else "wins_csv" if wins else "slower_than_csv",
               "csv_baseline_us": baseline[shape], "wins_csv": wins,
               "speedup": baseline[shape] / best["us"] if best else None,
               "evidence_kind": best["kind"] if best else None,
               "wins_csv_every_round": bool(best and best["kind"] == "repeat_median" and best["repeat"]["max_us"] < baseline[shape]),
               "confirmation_pending": confirmation_pending,
               "complete_repeat_batches": complete_names[(shape, best["kernelName"])] if best else 0,
               "pipeline": geometry[0] if geometry else None,
               "candidate_count": len(final[shape]), "correct_candidate_count": len(correct_candidates),
               "correct_repeat_candidate_count": len(repeated_candidates),
               "best_sources": best["sources"] if best else [],
               "discovery_single_pass_faster_than_repeat": bool(discovery_single and
                                                                (repeat_best is None or discovery_single["us"] < repeat_best["us"]))}
        for label, item in (("discovery_best", discovery), ("discovery_single_pass_best", discovery_single)):
            row[label] = ({"kernelId": item["kernelId"], "kernelName": item["kernelName"],
                           "us": item["us"], "errRatio": item["errRatio"], "evidence_kind": item["kind"],
                           "wins_csv": item["us"] < baseline[shape],
                           "confirmation_pending": item["kind"] == "single_pass" or (shape, item["kernelName"]) in pending_names,
                           "sources": item["sources"]} if item else None)
        rows.append(row)
    losses = [row for row in rows if row["status"] in ("slower_than_csv", "no_correct_opus")]
    pending = [row for row in rows if row["status"] == "pending"]
    nk, mc = defaultdict(list), defaultdict(list)
    for row in rows:
        nk[f'N{row["N"]}/K{row["K"]}'].append(row)
        mc[m_class(row["M"])].append(row)
    now = datetime.now(timezone.utc).isoformat()
    summary = {"generated_utc": now, "scope": "NT only", "baseline": str(Path(args.baseline).resolve()),
               "expected_shapes": len(expected), "completed_shapes": len(expected) - len(pending),
               "wins_csv": sum(row["wins_csv"] for row in rows), "losses_csv": len(losses),
               "pending_shapes": len(pending), "no_correct_opus": sum(row["status"] == "no_correct_opus" for row in rows),
               "repeat_median_best_shapes": sum(row["evidence_kind"] == "repeat_median" for row in rows),
               "repeat_median_wins_csv": sum(row["wins_csv"] and row["evidence_kind"] == "repeat_median" for row in rows),
               "single_pass_best_shapes": sum(row["evidence_kind"] == "single_pass" for row in rows),
               "single_pass_wins_csv": sum(row["wins_csv"] and row["evidence_kind"] == "single_pass" for row in rows),
               "selected_best_wins_csv_every_round": sum(row["wins_csv_every_round"] for row in rows),
               "confirmation_pending_best_shapes": sum(row["confirmation_pending"] for row in rows),
               "require_repeat_winners": require_repeat_winners,
               "repeat_winner_missing_shapes": sum(row["correct_repeat_candidate_count"] == 0 for row in rows),
               "discovery_best_wins_csv": sum(bool(row["discovery_best"] and row["discovery_best"]["wins_csv"]) for row in rows),
               "discovery_best_single_pass_shapes": sum(bool(row["discovery_best"] and row["discovery_best"]["evidence_kind"] == "single_pass") for row in rows),
               "discovery_single_pass_faster_than_repeat_shapes": sum(row["discovery_single_pass_faster_than_repeat"] for row in rows),
               "full_top8_completed_shapes": len(full_done), "full_top8_pending_shapes": len(expected) - len(full_done),
               "fallback_expected_shapes": len(fallback_expected) if args.fallback_expected else None,
               "fallback_completed_shapes": len(fallback_done & fallback_expected) if args.fallback_expected else len(fallback_done),
               "fallback_pending_shapes": len(fallback_expected - fallback_done) if args.fallback_expected else None,
               "full_validation_complete": bool(full_paths and full_done == set(expected)
                                                  and (not args.fallback_expected or fallback_expected.issubset(fallback_done))),
               "counts": dict(counts), "by_n_k": {group: summarize_group(items) for group, items in sorted(nk.items())},
               "by_m_class": {group: summarize_group(items) for group, items in mc.items()},
               "losses_by_n_k": dict(Counter(f'N{row["N"]}/K{row["K"]}' for row in losses).most_common()),
               "losses_by_m_class": dict(Counter(m_class(row["M"]) for row in losses)),
               "current_registry_filter": use_current_registry,
               "filter_to_expected": filter_to_expected,
               "current_registry_available_candidates": len(production_registry) if production_registry is not None else None,
               "excluded_experiment_candidate_count": len(excluded_candidates),
               "excluded_experiment_candidates": list(excluded_candidates.values()),
               "excluded_experiment_reasons": dict(Counter(row["reason"] for row in excluded_candidates.values())),
               "files": {"profile": list(map(str, profile_paths)), "progress": list(map(str, progress_paths)),
                         "full_progress": list(map(str, full_paths)), "fallback_progress": list(map(str, fallback_paths)),
                         "repeat": list(map(str, repeat_paths))},
               "missing_inputs": missing, "ignored_partial_final_lines": partial,
               "definitions": {"wins_csv": "Correct selected OPUS candidate us < the fixed CSV baseline; no same-round FlyDSL dependence",
                               "candidate_identity": "Exact six-field shape key plus complete current kernel name, with IDs checked for conflicts",
                               "single_pass": "Latest single observation of that exact candidate; all profile candidates participate",
                               "repeat_median": "Latest complete repeat batch; all expected rounds correct, automatic input rotation, median replaces all single observations",
                               "failed_repeat": "A complete batch containing any failed round invalidates that exact candidate",
                               "pending_repeat": "An incomplete batch cannot replace an existing complete median or single observation",
                               "require_repeat_winners": "When enabled, official best and CSV scoring use only correct complete automatic-rotation repeat medians; discovery stays separate",
                               "discovery_best": "Fastest correct candidate after completed-repeat replacement, including unconfirmed single-pass candidates; excluded from official scoring when repeats are required",
                               "full_top8_completed_shapes": "Only shapes in --full-progress, kept separate from earlier directed coverage",
                               "historical_results": "Excluded from current wins and candidate selection"}}
    return {"summary": summary, "shapes": rows, "remaining_losses": losses,
            "pending_shapes": pending, "repeat_batches": repeat_details}, final


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--expected", type=Path, default=DOC / "shapes_421.csv")
    parser.add_argument("--baseline", type=Path, default=DATA / "csv_baseline.csv")
    parser.add_argument("--profile", nargs="*", default=[], help="All-candidate OPUS/FlyDSL profile CSVs or quoted globs")
    parser.add_argument("--progress", nargs="*", default=[], help="Earlier directed progress JSONLs or quoted globs")
    parser.add_argument("--full-progress", nargs="*", default=[], help="Full421 Top8 progress JSONLs; also participate in selection")
    parser.add_argument("--fallback-progress", nargs="*", default=[], help="Known-winner Top0 supplement progress JSONLs")
    parser.add_argument("--fallback-expected", type=Path, help="Required supplement shape CSV for coverage accounting")
    parser.add_argument("--repeat", nargs="*", default=[], help="Automatic-rotation six-round repeat JSONLs or quoted globs")
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--err-ratio", type=float, default=.05)
    parser.add_argument("--output-dir", type=Path, default=LOCAL / "full421_report")
    parser.add_argument("--candidate-details", action="store_true", help="Also write every final candidate; may be a large CSV")
    parser.add_argument("--current-registry", action="store_true", help="CPU-filter isolated unavailable/name-mismatched/production-shape-illegal candidates")
    parser.add_argument("--filter-to-expected", action="store_true", help="For directed subsets, ignore broader input observations outside --expected")
    parser.add_argument("--require-repeat-winners", action="store_true", help="Official winners must have a correct complete repeat median; keep single-pass discovery separate")
    parser.add_argument("--workflow-state", type=Path, help="Update this workflow state JSON after writing report artifacts")
    parser.add_argument("--physical-gpus", default="0,1,2,3", help="Physical GPUs used by this batch")
    parser.add_argument("--finalize", action="store_true", help="Require complete full Top8/specified supplement coverage before marking step4 complete")
    args = parser.parse_args()
    if args.rounds < 2 or not 0 <= args.err_ratio <= 1:
        parser.error("Use rounds>=2 and err-ratio in [0,1]")
    if not any((args.profile, args.progress, args.full_progress, args.fallback_progress, args.repeat)):
        parser.error("Supply at least one --profile, --progress, --full-progress, --fallback-progress or --repeat input")
    try:
        physical_gpus = [int(value) for value in args.physical_gpus.split(",")]
        if not 1 <= len(physical_gpus) <= 4 or len(set(physical_gpus)) != len(physical_gpus) or any(gpu not in range(8) for gpu in physical_gpus):
            raise ValueError("Select one to four distinct physical GPUs from 0 to 7")
        report, candidates = analyze(args)
        summary, rows = report["summary"], report["shapes"]
        summary["physical_gpu_ids"] = physical_gpus
        if args.finalize and (not summary["full_validation_complete"] or summary["no_correct_opus"] or summary["missing_inputs"]):
            raise ValueError("--finalize requires complete full Top8/specified supplement coverage, a correct OPUS winner for every shape, and no missing input paths")
        if args.finalize and args.require_repeat_winners and (
                summary["repeat_median_best_shapes"] != summary["expected_shapes"]
                or summary["single_pass_best_shapes"] or summary["confirmation_pending_best_shapes"]):
            raise ValueError("--finalize with --require-repeat-winners requires a complete correct repeat winner for every expected shape, zero single-pass winners, and zero pending confirmations")
        output = args.output_dir.resolve()
        output.mkdir(parents=True, exist_ok=True)
        atomic_text(output / "report.json", json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        atomic_text(output / "summary.json", json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        atomic_csv(output / "shape_status.csv", rows, RESULT_FIELDS)
        atomic_csv(output / "best_opus.csv", [row for row in rows if row["us"] is not None], RESULT_FIELDS)
        atomic_csv(output / "remaining_losses.csv", report["remaining_losses"], RESULT_FIELDS)
        atomic_csv(output / "pending_shapes.csv", report["pending_shapes"], RESULT_FIELDS)
        atomic_csv(output / "confirmation_pending.csv", [row for row in rows if row["confirmation_pending"]], RESULT_FIELDS)
        atomic_csv(output / "discovery_unconfirmed.csv", [row for row in rows if row["discovery_single_pass_faster_than_repeat"]], RESULT_FIELDS)
        atomic_csv(output / "excluded_experiment_candidates.csv", summary["excluded_experiment_candidates"],
                   SHAPE_FIELDS + ("kernelId", "kernelName", "reason", "observations", "sources"))
        if args.candidate_details:
            details = ({**shape_row(shape), **record, "evidence_kind": record["kind"]}
                       for shape in sorted(candidates) for record in candidates[shape].values())
            atomic_csv(output / "candidate_status.csv", details,
                       CONFIG_FIELDS + ("correct", "evidence_kind", "sources", "repeat"))
        lines = [f'本轮累计完成 {summary["completed_shapes"]}/{summary["expected_shapes"]} 个 NT shape，'
                 f'胜 CSV {summary["wins_csv"]} 个，剩余落后 {summary["losses_csv"]} 个，待测 {summary["pending_shapes"]} 个。', "",
                 f'完整 Top8 首轮完成 {summary["full_top8_completed_shapes"]}/{summary["expected_shapes"]} 个；'
                 f'已有六轮中位证据的最佳结果 {summary["repeat_median_best_shapes"]} 个，'
                 f'其中 {summary["repeat_median_wins_csv"]} 个胜 CSV。'
                 f'最佳结果仍为单轮的 {summary["single_pass_best_shapes"]} 个，'
                 f'其中 {summary["single_pass_wins_csv"]} 个胜 CSV。', "",
                 'CSV 基线固定使用 handoff_data/csv_baseline.csv。完整复测覆盖同一候选的单轮结果；'
                 '完整复测中任一轮失败的候选失效。历史跨机器赢家没有进入本轮胜出统计。', ""]
        if summary["fallback_expected_shapes"] is not None:
            lines += [f'已知赢家补测完成 {summary["fallback_completed_shapes"]}/{summary["fallback_expected_shapes"]} 个 shape。', ""]
        if summary["current_registry_filter"]:
            lines += [f'生产候选筛选已启用；排除 {summary["excluded_experiment_candidate_count"]} 个不可用、'
                      f'名字不符或不在生产支持 shape 范围内的实验候选，'
                      f'共 {summary["counts"].get("excluded_experiment_observations", 0)} 条观测。', ""]
        if summary["require_repeat_winners"]:
            lines += ['正式胜负仅使用完整正确复测的中位耗时。单轮候选继续保留在 discovery_best，'
                      '不进入正式胜 CSV 统计。', "",
                      f'单轮候选快于已确认最佳或尚无已确认最佳的 shape：'
                      f'{summary["discovery_single_pass_faster_than_repeat_shapes"]} 个；'
                      f'尚无完整正确复测最佳的 shape：{summary["repeat_winner_missing_shapes"]} 个。', ""]
        if summary["losses_by_n_k"]:
            lines += ['剩余落后按 N/K：', ""] + [f'- {group}：{count} 个' for group, count in summary["losses_by_n_k"].items()] + [""]
        if summary["losses_by_m_class"]:
            lines += ['剩余落后按 M：', ""] + [f'- {group}：{count} 个' for group, count in summary["losses_by_m_class"].items()] + [""]
        lines += [f'正确性失败观测：单轮 {summary["counts"].get("single_pass_incorrect_measurements", 0)} 条，'
                  f'复测 {summary["counts"].get("repeat_incorrect_measurements", 0)} 条。'
                  f'无正确 OPUS 候选的已测 shape：{summary["no_correct_opus"]} 个。', ""]
        atomic_text(output / "summary_zh.md", "\n".join(lines))
        if args.workflow_state:
            state = json.loads(args.workflow_state.read_text()) if args.workflow_state.exists() else {}
            state.update(updated_utc=summary["generated_utc"], scope="NT only", gpu_ids=physical_gpus)
            state["latest_report"] = {"path": str(output / "report.json"),
                                      **{field: summary[field] for field in ("completed_shapes", "wins_csv", "losses_csv", "pending_shapes", "full_top8_completed_shapes", "confirmation_pending_best_shapes")}}
            if args.full_progress:
                state["current_step"] = 4
                state["phase"] = "full421 Top8 and known-winner supplements"
                state.setdefault("steps", {})["4"] = ("complete" if args.finalize else "in_progress")
            atomic_text(args.workflow_state, json.dumps(state, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        print(json.dumps({field: summary[field] for field in (
            "expected_shapes", "completed_shapes", "wins_csv", "losses_csv", "pending_shapes",
            "repeat_median_best_shapes", "repeat_median_wins_csv", "single_pass_best_shapes",
            "full_top8_completed_shapes", "fallback_completed_shapes", "full_validation_complete",
            "confirmation_pending_best_shapes", "no_correct_opus", "current_registry_filter",
            "excluded_experiment_candidate_count", "require_repeat_winners", "repeat_winner_missing_shapes",
            "discovery_best_single_pass_shapes", "discovery_single_pass_faster_than_repeat_shapes")}, indent=2))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
