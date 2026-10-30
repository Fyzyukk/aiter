#!/usr/bin/env python3
"""CPU-only step3 inputs from current step1/2 and balanced repeat evidence.

Historical results supply source labels and pending inputs only. A completed
six-round candidate uses its median and must pass every round. Its median
replaces that same candidate's single-pass observation; un-repeated candidates
remain explicitly single-pass observations. No cross-machine win union is used.
"""

import argparse
import csv
import glob
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from audit_progress import DOC, DATA, ROOT, SHAPE_FIELDS, csv_rows, key, m_class, name_geometry, pure_tuner, write_csv

LOCAL = Path(__file__).resolve().parent
GEO_FIELDS = ("M", "N", "K", "pipeline", "BM", "BN", "BK", "TM", "TN")


def finite(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def correct(record, limit):
    us, err = finite(record.get("us")), finite(record.get("errRatio"))
    return us is not None and us > 0 and err is not None and 0 <= err <= limit and record.get("correct", True)


def jsonl(path, missing, partial):
    path = Path(path)
    if not path.exists():
        missing.append(str(path))
        return
    with path.open() as stream:
        for line, text in enumerate(stream, 1):
            if not text.strip():
                continue
            try:
                record = json.loads(text)
            except json.JSONDecodeError:
                partial.append(f"{path}:{line}")
                continue
            yield record, {"path": str(path), "line": line, "completed_utc": record.get("completed_utc", "")}


def classes(rows):
    return {"by_n_k": dict(Counter(f'N{row["N"]}/K{row["K"]}' for row in rows).most_common()),
            "by_m_class": dict(Counter(m_class(row["M"]) for row in rows))}


def csv_safe(rows):
    return [{field: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
             for field, value in row.items()} for row in rows]


def geometry_cost(geometry, shape):
    _, bm, bn, _, _, _ = geometry
    m, n = shape[2:4]
    tiles = math.ceil(m / bm) * math.ceil(n / bn)
    padding = math.ceil(m / bm) * bm / m * math.ceil(n / bn) * bn / n
    occupancy = tiles / max(shape[1], 1)
    return padding - 1 + 2 * max(0, 1 - occupancy) + .1 * (math.ceil(occupancy) / occupancy - 1) + 8 / bm + 8 / bn


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--progress", type=Path, nargs="+", help="Current step1/2 JSONL files; defaults to local step1_progress/step2_progress")
    parser.add_argument("--repeat", type=Path, nargs="+", help="Current six-round repeat JSONL files; defaults to local step1_repeat*.jsonl and step2_repeat*.jsonl")
    parser.add_argument("--output-dir", type=Path, default=LOCAL / "step3_inputs")
    parser.add_argument("--err-ratio", type=float, default=.05)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--representatives-per-group", type=int, default=2)
    args = parser.parse_args()
    if not 0 <= args.err_ratio <= 1 or args.rounds < 2 or args.representatives_per_group < 1:
        parser.error("Use err-ratio in [0,1], rounds>=2 and at least one representative per group")
    current_paths = args.progress or [LOCAL / "step1_progress.jsonl", LOCAL / "step2_progress.jsonl"]
    repeat_paths = args.repeat or [Path(p) for pattern in ("step1_repeat*.jsonl", "step2_repeat*.jsonl")
                                  for p in sorted(glob.glob(str(LOCAL / pattern)))]
    baseline = {key(row): float(row["us"]) for row in csv_rows(DATA / "csv_baseline.csv")}
    expected_by_step = {name: {key(row) for row in csv_rows(DATA / filename)}
                        for name, filename in (("step1", "m_repeat_32_shapes.csv"), ("step2", "regressions_46_shapes.csv"))}
    expected = set().union(*expected_by_step.values())
    if len(expected) != 76:
        raise ValueError(f"Expected 76 unique step1/2 shapes, got {len(expected)}")
    missing, partial, current_events = [], [], defaultdict(list)
    # Keep the final event for each source file/shape. Different files may have
    # searched different geometry sets; retain each file's candidate discovery.
    latest = {}
    for path in current_paths:
        for event, source in jsonl(path, missing, partial):
            shape = key(event["shape"])
            if shape not in expected:
                raise ValueError(f"Current progress contains an unexpected shape {shape} at {source}")
            current_events[shape].append({"event": event, "source": source})
            latest[(str(path), shape)] = (event, source)
    candidates = defaultdict(dict)
    for (_, shape), (event, source) in latest.items():
        result = event.get("best_opus")
        if result:
            candidate = {**result, "correct": correct(result, args.err_ratio), "kind": "single_pass",
                         "sources": [source], "rounds": None}
            name = result["kernelName"]
            old = candidates[shape].get(name)
            if old is None or source["completed_utc"] >= old["sources"][-1]["completed_utc"]:
                if old:
                    candidate["sources"] = old["sources"] + candidate["sources"]
                candidates[shape][name] = candidate
    repeated = defaultdict(dict)
    excluded_repeat, all_repeat_events = [], []
    for path in repeat_paths:
        for record, source in jsonl(path, missing, partial):
            shape = key(record["shape"])
            if shape not in expected:
                raise ValueError(f"Repeat contains an unexpected shape {shape} at {source}")
            if int(record.get("rotate_args", 0)) != 0:
                excluded_repeat.append({"reason": "fixed/hot-cache rotation is not comparable to CSV", **source})
                continue
            name = record["kernelName"]
            round_index = int(record["round"])
            all_repeat_events.append({"record": record, "source": source})
            repeated[(shape, name)][round_index] = {"record": record, "source": source}
    repeat_status = defaultdict(list)
    for (shape, name), rounds in repeated.items():
        complete = set(range(args.rounds)).issubset(rounds)
        used = [rounds[i] for i in range(args.rounds) if i in rounds]
        valid = [entry["record"]["us"] for entry in used if correct(entry["record"], args.err_ratio)]
        all_correct = complete and len(valid) == args.rounds
        row = {"kernelName": name, "rounds_seen": sorted(rounds), "complete": complete,
               "all_rounds_correct": all_correct, "median_us": statistics.median(valid) if valid else None,
               "min_us": min(valid) if valid else None, "max_us": max(valid) if valid else None,
               "pstdev_us": statistics.pstdev(valid) if len(valid) > 1 else 0 if valid else None,
               "rounds": [{**entry["record"], "source": entry["source"]} for entry in used]}
        repeat_status[shape].append(row)
        if complete:
            candidates[shape][name] = {"kernelName": name, "kernelId": used[0]["record"].get("kid"),
                                      "us": row["median_us"], "errRatio": max(finite(entry["record"].get("errRatio")) if finite(entry["record"].get("errRatio")) is not None else 1 for entry in used),
                                      "correct": all_correct, "kind": "repeat_median",
                                      "sources": [entry["source"] for entry in used], "rounds": row}
    # History is provenance only; never add it to candidates or current wins.
    history = defaultdict(list)
    historical_geometry = defaultdict(set)
    remote = {key(row): row for row in csv_rows(DOC / "all_shapes_421.csv")}
    for shape, row in remote.items():
        historical_geometry[shape].add((row["best_opus_pipeline"], *map(int, row["best_opus_tile"].split("x")),
                                       *map(int, row["best_opus_waves"].split("x"))))
    for filename in ("local_full185_progress.jsonl", "local_priority22_progress.jsonl"):
        source_latest = {}
        for event, source in jsonl(DATA / filename, missing, partial):
            source_latest[key(event["shape"])] = (event, source)
        for shape, (event, source) in source_latest.items():
            result = event.get("best_opus")
            ok = bool(result and correct(result, args.err_ratio))
            us = float(result["us"]) if ok else None
            wins = bool(ok and us < baseline[shape])
            history[shape].append({"kind": "historical_local", "source": source, "us": us,
                                   "wins_csv": wins, "loss_csv": not wins,
                                   "kernelName": result.get("kernelName") if result else None})
            geo = name_geometry(result["kernelName"]) if result else None
            if geo:
                historical_geometry[shape].add(geo)
    for row in csv_rows(DATA / "both_losses_67_shapes.csv"):
        history[key(row)].append({"kind": "handoff_both_losses67", "source": str(DATA / "both_losses_67_shapes.csv"),
                                  "wins_csv": False, "loss_csv": True, "us": None})
    current = {}
    for shape in expected:
        observed = bool(current_events.get(shape)) or bool(repeat_status.get(shape))
        valid = [record for record in candidates[shape].values() if record["correct"]]
        best = min(valid, key=lambda record: float(record["us"])) if valid else None
        wins = bool(best and float(best["us"]) < baseline[shape])
        complete_repeat = any(record["complete"] for record in repeat_status.get(shape, []))
        incomplete_repeat = any(not record["complete"] for record in repeat_status.get(shape, []))
        row = {**dict(zip(SHAPE_FIELDS, shape)), "status": "pending" if not observed else "observed_win" if wins else "observed_loss",
               "observed": observed, "wins_csv": wins, "csv_baseline_us": baseline[shape],
               "opus_us": float(best["us"]) if best else None,
               "speedup": baseline[shape] / float(best["us"]) if best else None,
               "kernelName": best["kernelName"] if best else None,
               "kind": best["kind"] if best else None,
               "confirmation_pending": incomplete_repeat,
               "has_complete_repeat": complete_repeat,
               "best_sources": best["sources"] if best else [], "historical_sources": history.get(shape, [])}
        current[shape] = row
    current_losses = [row for row in current.values() if row["status"] == "observed_loss"]
    history_loss_shapes = {shape for shape, evidence in history.items() if any(record["loss_csv"] for record in evidence)}
    pending = []
    for shape in sorted(history_loss_shapes):
        if current.get(shape, {}).get("observed"):
            continue
        row = {**dict(zip(SHAPE_FIELDS, shape)), "status": "historical_loss_pending_current_measurement",
               "csv_baseline_us": baseline[shape], "historical_sources": history[shape],
               "historical_mixed_evidence": any(record["wins_csv"] for record in history[shape])}
        pending.append(row)
    followup = sorted(current_losses + pending, key=lambda row: (row["N"], row["K"], row["M"]))
    groups = defaultdict(list)
    for row in followup:
        groups[(row["N"], row["K"], m_class(row["M"]))].append(row)
    representatives = []
    for (n, k, mc), rows in sorted(groups.items()):
        # The slowest relative observation plus largest M cover the current
        # bottleneck and the long-running tail. Pending groups use median M
        # plus largest M rather than inventing current performance evidence.
        observed_rows = [row for row in rows if row["status"] == "observed_loss"]
        first = max(observed_rows, key=lambda row: (1 / row["speedup"] if row.get("speedup") else float("inf"), row["M"])) if observed_rows else rows[len(rows) // 2]
        ordered = [first, max(rows, key=lambda row: row["M"]), min(rows, key=lambda row: row["M"])]
        chosen = []
        for row in ordered + rows:
            if row not in chosen:
                chosen.append(row)
            if len(chosen) == min(args.representatives_per_group, len(rows)):
                break
        for row in chosen:
            representatives.append({**row, "group_n": n, "group_k": k, "group_m_class": mc,
                                    "group_shape_count": len(rows),
                                    "selection_reason": "worst current slowdown / largest M" if observed_rows else "historical pending median / largest M"})
    common, tuner = pure_tuner()
    registry = common.get_registry("a8w8_blockscale_bpreshuffle", "gfx950", 256)
    name_to_instance = {instance.name: instance for instance in registry.values()}
    for shape, records in candidates.items():
        for name in records:
            if name not in name_to_instance:
                raise ValueError(f"Current candidate name is unavailable in this registry: {shape} {name}")
    geometry = tuner["_opus_geometry"]
    known_geometries, exploratory_geometries, unavailable_geometries, geometry_details = [], [], [], []
    representative_keys = {key(row) for row in representatives}
    for row in followup:
        shape = key(row)
        m, n, k = shape[2:5]
        geometry_counts = Counter(geometry(instance) for instance in registry.values()
                                  if common.a8w8_mxscale_bpreshuffle_supports_shape(instance, m, n, k))
        legal = set(geometry_counts)
        requested = set(historical_geometry.get(shape, set()))
        name = current.get(shape, {}).get("kernelName")
        if name in name_to_instance:
            requested.add(geometry(name_to_instance[name]))
        for geo in sorted(requested):
            record = dict(zip(GEO_FIELDS, (m, n, k, *geo)))
            if geo in legal:
                known_geometries.append(record)
                geometry_details.append({**record, "selection": "saved remote/local or current best geometry",
                                         "legal_candidate_count": geometry_counts[geo], "shape_status": row["status"]})
            else:
                unavailable_geometries.append({**record, "reason": "historical geometry unavailable/illegal for current shape"})
        if shape in representative_keys:
            selected = set(requested & legal)
            for pipeline in ("prefetch_b_vgpr", "prefetch_b_vgpr_overlap"):
                if n in (768, 1536, 2048, 6144):
                    for bn in (64, 128):
                        for waves in (2, 4):
                            options = [geo for geo in legal if geo[0] == pipeline and geo[2] == bn and geo[4] * geo[5] == waves]
                            if options:
                                selected.add(min(options, key=lambda geo: (geo not in requested, geometry_cost(geo, shape), geo)))
                else:
                    options = [geo for geo in legal if geo[0] == pipeline]
                    if options:
                        selected.add(min(options, key=lambda geo: (geo not in requested, geometry_cost(geo, shape), geo)))
            if n >= 6144:
                for pipeline in ("pin_agpr", "pingpong_lds"):
                    options = sorted((geo for geo in legal if geo[0] == pipeline), key=lambda geo: (geo not in requested, geometry_cost(geo, shape), geo))
                    selected.update(options[:1 if pipeline == "pin_agpr" else 2])
            exploratory_geometries.extend(dict(zip(GEO_FIELDS, (m, n, k, *geo))) for geo in sorted(selected))
            geometry_details.extend({**dict(zip(GEO_FIELDS, (m, n, k, *geo))), "selection": "representative targeted comparison",
                                     "legal_candidate_count": geometry_counts[geo], "shape_status": row["status"]} for geo in sorted(selected))
    observed_rows = [row for row in current.values() if row["observed"]]
    summary = {"fixed_baseline": str(DATA / "csv_baseline.csv"), "correctness_limit": args.err_ratio,
               "current_progress_files": [str(path) for path in current_paths], "current_repeat_files": [str(path) for path in repeat_paths],
               "expected_unique_shapes": len(expected), "completed_unique_shapes": len(observed_rows),
               "wins_csv": sum(row["wins_csv"] for row in observed_rows), "observed_losses_csv": len(current_losses),
               "pending_current76": len(expected) - len(observed_rows),
               "best_is_repeat_median": sum(row["kind"] == "repeat_median" for row in observed_rows),
               "best_is_single_pass": sum(row["kind"] == "single_pass" for row in observed_rows),
               "repeat_confirmation_pending_shapes": sum(row["confirmation_pending"] for row in observed_rows),
               "historical_loss_pending_shapes": len(pending), "historical_loss_pending_mixed_evidence": sum(row["historical_mixed_evidence"] for row in pending),
               "historical_losses_removed_by_current_win": sum(current.get(shape, {}).get("wins_csv", False) for shape in history_loss_shapes),
               "observed_loss_classes": classes(current_losses), "historical_pending_classes": classes(pending),
               "followup_shapes": len(followup), "representative_shapes": len(representatives),
               "known_geometry_rows": len(known_geometries), "representative_geometry_rows": len(exploratory_geometries),
               "known_geometry_candidates": sum(row["legal_candidate_count"] for row in geometry_details if row["selection"] != "representative targeted comparison"),
               "representative_geometry_candidates": sum(row["legal_candidate_count"] for row in geometry_details if row["selection"] == "representative targeted comparison"),
               "unavailable_geometry_rows": len(unavailable_geometries), "missing_files": missing,
               "ignored_partial_lines": partial, "excluded_repeat_measurements": excluded_repeat,
               "step_membership": {step: {"expected": len(shapes), "completed": sum(current[shape]["observed"] for shape in shapes),
                                          "wins_csv": sum(current[shape]["wins_csv"] for shape in shapes)} for step, shapes in expected_by_step.items()},
               "selection_rule": "Latest observation per source file/shape; each complete six-round candidate replaces that same candidate with its all-correct median; select fastest remaining current candidate. Unrepeated candidates retain single_pass label.",
               "historical_rule": "Historical loss union only supplies pending followup shapes. Historical wins are never current candidates or current wins."}
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    details = {"summary": summary, "current76": list(current.values()),
               "current_events": [{"shape": dict(zip(SHAPE_FIELDS, shape)), "events": records} for shape, records in current_events.items()],
               "repeat_events": all_repeat_events,
               "candidates": [{"shape": dict(zip(SHAPE_FIELDS, shape)), "candidates": list(records.values()),
                                "repeat_evidence": repeat_status.get(shape, [])} for shape, records in candidates.items()]}
    (out / "evidence.json").write_text(json.dumps(details, indent=2, ensure_ascii=False) + "\n")
    status_fields = SHAPE_FIELDS + ("status", "opus_us", "csv_baseline_us", "speedup", "kind", "kernelName", "confirmation_pending", "has_complete_repeat", "best_sources", "historical_sources")
    write_csv(out / "current76_status.csv", csv_safe(sorted(current.values(), key=key)), status_fields)
    write_csv(out / "observed_losses_shapes.csv", current_losses, SHAPE_FIELDS)
    write_csv(out / "observed_losses_status.csv", csv_safe(current_losses), status_fields)
    write_csv(out / "historical_pending_losses_shapes.csv", pending, SHAPE_FIELDS)
    write_csv(out / "historical_pending_losses_status.csv", csv_safe(pending), SHAPE_FIELDS + ("status", "csv_baseline_us", "historical_mixed_evidence", "historical_sources"))
    write_csv(out / "followup_shapes.csv", followup, SHAPE_FIELDS)
    write_csv(out / "followup_geometries.csv", known_geometries, GEO_FIELDS)
    write_csv(out / "representative_shapes.csv", representatives, SHAPE_FIELDS)
    write_csv(out / "representative_status.csv", csv_safe(representatives), status_fields + ("group_m_class", "group_shape_count", "selection_reason"))
    write_csv(out / "representative_geometries.csv", exploratory_geometries, GEO_FIELDS)
    write_csv(out / "geometry_selection_details.csv", geometry_details, GEO_FIELDS + ("selection", "legal_candidate_count", "shape_status"))
    write_csv(out / "unavailable_geometries.csv", unavailable_geometries, GEO_FIELDS + ("reason",))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
