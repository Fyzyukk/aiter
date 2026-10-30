#!/usr/bin/env python3
"""CPU-only stage3 profile analysis and alternated-repeat shortlist.

Worker candidates/*.json contain counts, not per-kernel timings. Completed
profile.csv supplies timings; progress.jsonl supplies interim shape winners.
No torch/aiter import and no compilation or GPU use occurs in this program.
"""

import argparse
import copy
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

from prepare_opus_repeat import (LOCAL, ROOT, DATA, KEYS, csv_rows, key,
                                 number, pure_common, registry_metadata,
                                 shape_row, traits_args, write_csv)
from prepare_co_experiment import B_FIELDS, b_metadata, pin_wrapper
from inspect_co_parents import peer_key

B_PIPES = ("prefetch_b_vgpr", "prefetch_b_vgpr_overlap")
# Canonical 17 base parameters from the current C++ template defaults. These
# are completed before pairing abbreviated ordinary-B JIT traits with overlap.
B_DEFAULTS = (None, None, None, None, "0", "false", "7168", "4", "1",
              "0", "2", "0", "256", "false", "false", "false", "0")


def canonical_b(entry):
    if isinstance(entry["traits"], dict):
        args = tuple(str(v).lower() for v in entry["traits"]["args"])
    else:
        parsed = traits_args(entry["traits"])
        if parsed is None:
            raise ValueError(f"Unparseable B traits: {entry['traits']}")
        args = parsed[1]
    if not 4 <= len(args) <= 18:
        raise ValueError(f"Unexpected B trait count: {len(args)}")
    base = args[:17] + B_DEFAULTS[len(args[:17]):]
    publish = int(args[17]) if len(args) == 18 else 0 if entry["pipeline"] == B_PIPES[1] else -1
    return base, publish


def metadata_details(instance, meta):
    entry = instance.bpreshuffle_co or instance.bpreshuffle_entry
    info = {"pipeline": entry["pipeline"], "kind": meta["kind"],
            "kid": meta["kid"], "kernelName": meta["kernelName"],
            "geometry": meta["geometry"], "traits": meta["traits"]}
    if entry["pipeline"] in B_PIPES:
        args, publish = canonical_b(entry)
        info["actual_schedule"] = b_metadata(args)
        info["publish_m"] = publish
    elif entry["pipeline"] == "pin_agpr":
        _, wrapper = pin_wrapper(entry)
        info.update(panel=int(entry["traits"]["args"][0]),
                    unroll=int(entry["traits"]["args"][1]),
                    pad_m=entry["pad_m"], wide_output=entry["wide_output"],
                    group_m=int(wrapper[5]))
    elif entry["pipeline"] == "pingpong_lds":
        args = traits_args(entry["traits"])[1]
        info.update(panel=int(args[2]), fixed_k=int(args[3]), group_m=int(args[5]),
                    wave_m=int(args[4]), cache_a=int(args[6]), cache_b=int(args[7]),
                    frag=args[8] == "true", xcd=args[9] == "true",
                    bounded_output=args[10] == "true", publish_m=int(args[11]),
                    group_n=int(args[12]))
    return info


def strict_peers(registry):
    groups = defaultdict(lambda: defaultdict(list))
    for kid, instance in registry.items():
        entry = instance.bpreshuffle_co or instance.bpreshuffle_entry
        if entry["pipeline"] not in B_PIPES:
            continue
        if instance.bpreshuffle_co:
            identity = ("CO", peer_key(entry))
        else:
            base, _ = canonical_b(entry)
            # Both JIT routes use the common compile flags and same kernel
            # launch convention; include every layout and non-trait contract.
            identity = ("JIT", base, instance.B_M, instance.B_N, instance.B_K,
                        instance.T_M, instance.T_N,
                        entry.get("m_align"), tuple(entry.get("headers", ())),
                        entry.get("grid_layout"))
        groups[identity][entry["pipeline"]].append(kid)
    result = {}
    for pipes in groups.values():
        for pipeline, kids in pipes.items():
            other = B_PIPES[1] if pipeline == B_PIPES[0] else B_PIPES[0]
            for kid in kids:
                matches = pipes.get(other, [])
                # Ordinary -1 versus overlap0 is the intended strict schedule
                # pair. Other measured overlap publish points still add their
                # ordinary-B counterpart when they enter the shortlist.
                result[kid] = [peer for peer in matches
                               if pipeline != B_PIPES[0] or
                               canonical_b(registry[peer].bpreshuffle_co or registry[peer].bpreshuffle_entry)[1] == 0]
    return result


def read_json_lines(path):
    records, invalid = [], []
    if not path.exists():
        return records, invalid
    for index, line in enumerate(path.read_text().splitlines(), 1):
        try:
            records.append(json.loads(line))
        except ValueError:
            invalid.append({"path": str(path), "line": index})
    return records, invalid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=LOCAL / "step3_existing_round1")
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--shapes", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--err-ratio", type=float, default=.05)
    parser.add_argument("--write-repeat", action="store_true")
    parser.add_argument("--include-experiment-parents", action="store_true",
                        help="Require experimental manifest points and explicitly include each legal new CO plus its exact original parent")
    args = parser.parse_args()
    if args.topk < 1:
        parser.error("topk must be positive")
    profile = args.profile or args.run_dir / "profile.csv"
    shapes_path = args.shapes or args.run_dir / "shapes.csv"
    output = args.output_dir or args.run_dir / "cpu_analysis"
    shapes = sorted({key(row) for row in csv_rows(shapes_path)})
    baseline = {key(row): float(row["us"]) for row in csv_rows(DATA / "csv_baseline.csv")}
    common = pure_common()
    registry = common.get_registry("a8w8_blockscale_bpreshuffle", "gfx950", 256)
    metadata, guard_pairs = registry_metadata()
    peers = strict_peers(registry)
    manifest_path = ROOT / "csrc/opus_gemm/gen_co/co_kernels_gfx950.json"
    manifest = json.loads(manifest_path.read_text())
    experiment_points = manifest.get("_experimental_extensions", {}).get("points", [])
    integrated_points = manifest.get("_integrated_extensions", {}).get("points", [])
    active_points = [(point, "experimental") for point in experiment_points] + [(point, "integrated") for point in integrated_points]
    if args.include_experiment_parents and not active_points:
        parser.error("--include-experiment-parents requires experimental or integrated manifest extension points")
    for point, _ in active_points:
        new_kid, seed_kid = int(point["new_kid"]), int(point["seed_kid"])
        if new_kid not in metadata or seed_kid not in metadata or not registry[new_kid].bpreshuffle_co or not registry[seed_kid].bpreshuffle_co:
            parser.error(f"Experimental new/parent CO identity missing from current registry: {new_kid}/{seed_kid}")
    observations = defaultdict(dict)
    invalid = []
    if profile.exists():
        for line, row in enumerate(csv_rows(profile), 2):
            if row.get("libtype") != "opus" or key(row) not in shapes:
                continue
            shape, kid = key(row), int(row["kernelId"])
            us, err = number(row.get("us")), number(row.get("errRatio"))
            if kid not in registry:
                raise ValueError(f"Unavailable current kid {kid}, profile line {line}")
            correct = us is not None and us > 0 and err is not None and 0 <= err <= args.err_ratio
            if correct and row["kernelName"] != registry[kid].name:
                raise ValueError(f"Name/ID mismatch for kid {kid}, profile line {line}")
            if correct and not common.a8w8_mxscale_bpreshuffle_supports_shape(registry[kid], *shape[2:5]):
                raise ValueError(f"Shape contract fails for measured kid {kid}: {shape}")
            observations[shape][kid] = {"kid": kid, "us": us, "errRatio": err,
                                       "correct": correct, "source_line": line}
    progress, invalid_lines = read_json_lines(args.run_dir / "progress.jsonl")
    progress_latest = {key(row["shape"]): row for row in progress}
    worker_counts = []
    for path in sorted((args.run_dir / "candidates").glob("*.json")):
        try:
            worker_counts.append(json.loads(path.read_text()))
        except ValueError:
            invalid_lines.append({"path": str(path), "line": None})
    report_shapes, pipeline_rows, plan_shapes = [], [], []
    for shape in shapes:
        records = observations[shape]
        valid = sorted((r for r in records.values() if r["correct"]), key=lambda r: (r["us"], r["kid"]))
        by_pipe_kind = defaultdict(list)
        for observation in valid:
            info = metadata_details(registry[observation["kid"]], metadata[observation["kid"]])
            item = {**info, **observation, "csv_baseline_us": baseline[shape],
                    "wins_csv": observation["us"] < baseline[shape],
                    "strict_base_trait_peers": peers.get(observation["kid"], [])}
            by_pipe_kind[(info["pipeline"], info["kind"])].append(item)
        best = valid[0] if valid else None
        latest = progress_latest.get(shape)
        live_best = latest.get("best_opus") if latest else None
        shape_info = {**shape_row(shape), "csv_baseline_us": baseline[shape],
                      "profile_candidates": len(records), "correct_profile_candidates": len(valid),
                      "progress_complete": latest is not None,
                      "profile_best": best, "progress_best": live_best,
                      "wins_csv": bool(best and best["us"] < baseline[shape]) if records else
                      bool(live_best and live_best["us"] < baseline[shape]) if latest else None,
                      "pipeline_best": {}, "all_4wave_parents": [], "missing_legal_4wave_kids": []}
        selected = {r["kid"] for r in valid[:args.topk]}
        roles = defaultdict(set)
        for kid in selected:
            roles[kid].add("overall_topk")
        for (pipeline, kind), ordered in sorted(by_pipe_kind.items()):
            label = pipeline + ":" + kind
            shape_info["pipeline_best"][label] = ordered[:args.topk]
            for item in ordered[:args.topk]:
                pipeline_rows.append({**shape_row(shape), **item})
            chosen = ordered if pipeline == "pin_agpr" else ordered[:1]
            for item in chosen:
                selected.add(item["kid"])
                roles[item["kid"]].add("all_4wave_parents" if pipeline == "pin_agpr" else "best_pipeline_kind")
            if pipeline == "pin_agpr":
                shape_info["all_4wave_parents"] = ordered
        for pipeline in {p for p, _ in by_pipe_kind}:
            combined = sorted((item for (p, _), values in by_pipe_kind.items() if p == pipeline
                               for item in values), key=lambda item: (item["us"], item["kid"]))
            for item in combined[:args.topk]:
                selected.add(item["kid"])
                roles[item["kid"]].add("pipeline_topk")
        for kid, instance in registry.items():
            entry = instance.bpreshuffle_co
            if entry and entry["pipeline"] == "pin_agpr" and common.a8w8_mxscale_bpreshuffle_supports_shape(instance, *shape[2:5]):
                if kid not in records:
                    shape_info["missing_legal_4wave_kids"].append(kid)
                # Even failed parent observations must be tested explicitly in
                # the alternated correctness check rather than silently removed.
                selected.add(kid)
                roles[kid].add("all_legal_4wave_parents")
        # Experiments must measure every new point against its own untouched
        # parent in the same alternated rounds. Performance TopK cannot decide
        # whether an experiment deserves a correctness/paired comparison.
        for point, extension_kind in active_points:
            new_kid, seed_kid = int(point["new_kid"]), int(point["seed_kid"])
            if common.a8w8_mxscale_bpreshuffle_supports_shape(registry[new_kid], *shape[2:5]):
                if not common.a8w8_mxscale_bpreshuffle_supports_shape(registry[seed_kid], *shape[2:5]):
                    raise ValueError(f"Legal new CO {new_kid} has illegal parent {seed_kid} for {shape}")
                selected.update((new_kid, seed_kid))
                roles[new_kid].add(extension_kind + "_co")
                roles[seed_kid].add("exact_experiment_parent" if extension_kind == "experimental" else "exact_integrated_parent")
        for kid in list(selected):
            for other in peers.get(kid, []) + guard_pairs.get(kid, []):
                if common.a8w8_mxscale_bpreshuffle_supports_shape(registry[other], *shape[2:5]):
                    selected.add(other)
                    roles[other].add("strict_b_overlap_peer" if other in peers.get(kid, []) else "exact_bounded_output_peer")
        if records:
            candidates = []
            for kid in sorted(selected, key=lambda k: (records.get(k, {}).get("us") or math.inf, k)):
                observed = records.get(kid, {"kid": kid, "us": None, "errRatio": None, "correct": False})
                candidates.append({**copy.deepcopy(metadata[kid]), **observed,
                                   "roles": sorted(roles[kid]), "exact_opposite_kids": guard_pairs.get(kid, []),
                                   "strict_base_trait_peers": peers.get(kid, [])})
            plan_shapes.append({"shape": shape_row(shape), "csv_baseline_us": baseline[shape],
                                "remote_winner": None, "candidates": candidates})
        report_shapes.append(shape_info)
    summary = {"expected_shapes": len(shapes), "completed_shapes": sum(r["progress_complete"] or r["profile_candidates"] > 0 for r in report_shapes),
               "wins_csv": sum(r["wins_csv"] is True for r in report_shapes),
               "observed_losses": sum(r["wins_csv"] is False for r in report_shapes),
               "pending_shapes": sum(r["wins_csv"] is None for r in report_shapes),
               "profile_available": profile.exists(), "profile_candidates": sum(r["profile_candidates"] for r in report_shapes),
               "correct_profile_candidates": sum(r["correct_profile_candidates"] for r in report_shapes),
               "missing_legal_4wave_observations": sum(len(r["missing_legal_4wave_kids"]) for r in report_shapes),
               "repeat_shapes": len(plan_shapes), "repeat_candidates": sum(len(r["candidates"]) for r in plan_shapes),
               "experimental_points": len(experiment_points),
               "integrated_points": len(integrated_points),
               "experimental_candidate_shape_pairs": sum("experimental_co" in c["roles"] for s in plan_shapes for c in s["candidates"]),
               "experiment_parent_shape_pairs": sum("exact_experiment_parent" in c["roles"] for s in plan_shapes for c in s["candidates"]),
               "torch_imported": "torch" in sys.modules,
               "limitations": "Single-pass candidate discovery; alternated medians must confirm sorting. Worker JSON contains counts only, and is overwritten for each assigned shape."}
    output.mkdir(parents=True, exist_ok=True)
    (output / "analysis.json").write_text(json.dumps({"summary": summary, "shapes": report_shapes,
                                                      "worker_counts": worker_counts, "invalid_json_lines": invalid_lines}, indent=2, allow_nan=False) + "\n")
    if pipeline_rows:
        flat = []
        for row in pipeline_rows:
            flat.append({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v for k, v in row.items()})
        write_csv(output / "pipeline_top.csv", flat)
    if args.write_repeat:
        if not profile.exists() or len(plan_shapes) != len(shapes):
            parser.error("A completed profile covering every expected shape is required before writing repeat plan")
        plan = {"version": 2, "family": "a8w8_blockscale_bpreshuffle", "source_profile": str(profile),
                "metadata_module": "prepare_opus_repeat", "correctness_limit": args.err_ratio,
                "selection": "Overall and per-pipeline TopK, best pipeline CO/JIT, all legal 4wave parents, strict B/overlap base-trait peers (ordinary -1 vs overlap0), exact bounded-output peers, every legal experimental CO and its exact original parent",
                "experimental_points": experiment_points,
                "integrated_points": integrated_points,
                "shapes": plan_shapes}
        (output / "repeat_plan.json").write_text(json.dumps(plan, indent=2, allow_nan=False) + "\n")
        for shard in range(4):
            shard_plan = {**plan, "shapes": plan_shapes[shard::4]}
            (output / f"repeat_plan_shard{shard}.json").write_text(json.dumps(shard_plan, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
