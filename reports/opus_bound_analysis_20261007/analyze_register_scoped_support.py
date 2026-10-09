#!/usr/bin/env python3
"""CPU-only audit of complete scoped register support slices and claim epochs.

Only22 complete clean primary slices count toward646 runtime cases. Archived,
partial, polluted or identity-mismatched attempts do not count. Trace timing is
a screen, never final throughput or adoption evidence. Does not import HIP.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import subprocess

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def require(value, message):
    if not value:
        raise ValueError(message)


def json_lines(path):
    items = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                item = json.loads(line)
            except ValueError:
                continue
            if isinstance(item, dict):
                items.append(item)
    return items


def fingerprint(command, environment):
    """Match run_when_idle identity without calling its runtime entry point."""
    argv = command["argv"]
    output_index = argv.index("--output") + 1
    pending = [Path(value) for i, value in enumerate(argv) if i != output_index and Path(value).is_file()]
    pending.append(OUT / "experiment_runner.py")
    plan = read(Path(argv[argv.index("--plan") + 1]))
    pending.extend(Path(value) for value in plan.get("libraries", {}).values())
    files = {str(path.resolve()): sha(path) for path in pending if path.is_file()}
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    diff = subprocess.check_output(["git", "diff", "HEAD", "--", "csrc/opus_gemm"], cwd=ROOT)
    record = {"argv": argv, "env": environment, "files": files, "source_head": source,
              "source_diff_sha256": hashlib.sha256(diff).hexdigest()}
    return hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT / "results/register_scoped_support_analysis.json")
    parser.add_argument("--claim-log", type=Path, action="append",
                        help="claim log to inspect; may repeat. Each epoch must fit one complete log; default checks scoped and original logs")
    parser.add_argument("--screen-slowdown-percent", type=float, default=5.0,
                        help="descriptive profiler screen listing only, not an adoption gate")
    args = parser.parse_args()
    require(args.screen_slowdown_percent >= 0, "negative screen list threshold")
    manifest = read(OUT / "register_scoped_support_regression.json")
    queue = read(OUT / "register_scoped_idle_queue.json")
    inventory = read(OUT / "shape_inventory.json")
    scoped = read(OUT / "register_reuse_scoped/device_audit.json")
    build = read(OUT / "register_reuse_scoped/build_manifest.json")
    require(scoped["status"] == "scoped_aliases_match_tested_machine_code", "scoped ISA identity not verified")
    libraries = {entry["mode"]: {"path": str(OUT / "register_reuse_scoped" / entry["mode"] / "experiments.so"),
                                 "sha256": entry["binary_sha256"]} for entry in scoped["libraries"]}
    for side, value in libraries.items():
        require(value["sha256"] == sha(Path(value["path"])), "scoped disk binary hash changed")
        require(value["sha256"] == next(x for x in build["builds"] if x["mode"] == side)["binary_sha256"],
                "scoped manifest differs from device audit")
    shapes = {(r["M"], r["N"], r["K"]): r for r in inventory["rows"]}
    claim_paths = args.claim_log or [OUT / "register_scoped_claim_log.jsonl", OUT / "gpu_claim_log.jsonl"]
    claim_logs = {str(path.resolve()): json_lines(path) for path in claim_paths if path.exists()}
    records = {str(OUT / shard["path"]): shard for shard in manifest["shards"]}
    commands = [command for command in queue["commands"] if "--plan" in command["argv"]
                and command["argv"][command["argv"].index("--plan") + 1] in records]
    require(len(commands) == len(manifest["shards"]) == 22, "expected22 exact support commands")
    parts, clean_rows, archives, expected_pairs = [], [], [], []
    for command in commands:
        argv = command["argv"]
        plan_path = Path(argv[argv.index("--plan") + 1])
        result_path = Path(argv[argv.index("--output") + 1])
        log_path = Path(command["log"])
        plan = read(plan_path)
        record = records[str(plan_path)]
        require(plan["workspace"] is False and len(plan["targets"]) == record["targets"], "plan size/workspace changed")
        require(1 <= len(plan["targets"]) <= 32, "unexpected support slice size")
        require(plan["libraries"] == {side: value["path"] for side, value in libraries.items()}, "plan library paths changed")
        expected_pairs.extend((target["kid"], *target["shape"]) for target in plan["targets"])
        part = {"name": command["name"], "parent_id": record["kid"], "expected_rows": record["targets"],
                "plan": str(plan_path.relative_to(ROOT)), "plan_sha256": sha(plan_path),
                "result": str(result_path.relative_to(ROOT)), "status": "not_run"}
        parts.append(part)
        for archived in sorted(result_path.parent.glob(result_path.stem + ".attempt_*.json")):
            try:
                previous = read(archived)
                archives.append({"path": str(archived.relative_to(ROOT)), "sha256": sha(archived),
                    "status": previous.get("status"), "contamination": previous.get("contamination"),
                    "included_in_support_coverage": False, "timings_usable_for_final_decision": False,
                    "reason": "archived attempt; only a verified complete current primary counts"})
            except (ValueError, OSError) as exc:
                archives.append({"path": str(archived.relative_to(ROOT)), "included_in_support_coverage": False,
                                 "reason": str(exc)})
        if not result_path.exists():
            continue
        try:
            result = read(result_path)
        except (ValueError, OSError) as exc:
            part.update(status="incomplete_or_unreadable", reason=str(exc))
            continue
        part.update(application_status=result.get("status"), observed_rows=len(result.get("rows", [])),
                    result_sha256=sha(result_path))
        if result.get("contamination") or result.get("status") == "interrupted_external_gpu_work":
            part.update(status="contaminated_excluded", contamination=result.get("contamination"),
                        included_in_support_coverage=False)
            continue
        if result.get("status") != "passed" or not result.get("finished"):
            part.update(status="incomplete_or_failed", included_in_support_coverage=False)
            continue
        # Pick one contiguous command epoch surrounding the result, not a
        # start/end from different retried runs sharing the same command name.
        epochs = []
        for log_source, source_claims in claim_logs.items():
            starts = [(i, c) for i, c in enumerate(source_claims) if c.get("event") == "start"
                      and c.get("command", {}).get("name") == command["name"] and c["time"] <= result["started"]]
            for start_index, start in starts:
                ends = [(i, c) for i, c in enumerate(source_claims[start_index + 1:], start_index + 1)
                        if c.get("event") == "end" and c.get("name") == command["name"]]
                if not ends:
                    continue
                end_index, end = ends[0]
                if end["time"] >= result["finished"]:
                    epochs.append((log_source, source_claims, start_index, start, end_index, end))
        if not epochs:
            part.update(status="awaiting_claim_completion", included_in_support_coverage=False)
            continue
        log_source, claims, start_index, start, end_index, end = max(epochs, key=lambda epoch: epoch[3]["time"])
        if end.get("contamination"):
            part.update(status="contaminated_excluded", claim_end=end, included_in_support_coverage=False)
            continue
        try:
            require(end["time"] >= result["finished"] and end["returncode"] == 0 and end["contamination"] is False,
                    "result does not fit one clean completion epoch")
            interval = claims[start_index:end_index + 1]
            require(not any(c.get("event") == "external_work_started" for c in interval), "external activity in epoch")
            require(not any(c.get("event") == "start" and c.get("command", {}).get("name") == command["name"]
                            for c in interval[1:]), "result crosses retried command epochs")
            require(start["command"] == command, "executed command differs from queue")
            require(start.get("fingerprint") == fingerprint(command, queue.get("env", {})), "source/plan/binary fingerprint changed")
            claim_index, claim = max(((i, c) for i, c in enumerate(claims[:start_index + 1])
                if c.get("event") == "claimed"), key=lambda pair: pair[1]["time"])
            require(result["gpu"]["pci_bdf"] == result["gpu"]["expected_pci_bdf"] == claim["gpu"]["bdf"], "physical GPU mismatch")
            require(result["gpu"]["HIP_VISIBLE_DEVICES"] == str(claim["gpu"]["hip_index"]), "HIP visible identity mismatch")
            require(result["source_head"] == inventory["source_head"], "source HEAD mismatch")
            require(result["plan_sha256"] == sha(plan_path) and result["plan"] == plan, "recorded plan mismatch")
            require(result["libraries"] == libraries, "recorded library SHA/path mismatch")
            require(result["runner_sha256"] == sha(OUT / "experiment_runner.py"), "runner changed")
            require(result["rounds"] == 1 and result["iters"] == 11 and result["warmup"] == 5, "screen protocol mismatch")
            require(not result["event_confirmation_requested"] and not result["profiling_only"]
                    and result["target_index"] is None and result["label_filter"] is None, "not a full numerical screen slice")
            require(len(result["rows"]) == len(plan["targets"]), "missing planned rows")
            require([row["target_index"] for row in result["rows"]] == list(range(len(plan["targets"]))), "row indexes missing/reordered")
            local, expected_logs = [], []
            for index, (row, target) in enumerate(zip(result["rows"], plan["targets"])):
                require(row["kid"] == target["kid"] == record["kid"] and row["shape"] == target["shape"], "row shape/parent mismatch")
                require(target["signed"] is True and isinstance(target["seed"], int), "signed input/seed mismatch")
                require(row["split"] == 1 and "event_confirmation" not in row, "unexpected split or Event")
                for side in ["baseline", "candidate"]:
                    check = row["correctness"][side]
                    require(check["repetitions"] == 8 and check["errRatio"] == 0 and check["repeatable"]
                        and check["output_guards"] and check["workspace_guards"], "signed8rep/guard/repeatability failure")
                require(len(row["timings"]) == 2 and [t["label"] for t in row["timings"]] == ["baseline", "candidate"], "expected one AB trace pair")
                for timing in row["timings"]:
                    require(timing["round"] == 0 and timing["errRatio"] == 0 and timing["output_guards"]
                        and timing["workspace_guards"] and math.isfinite(timing["us"]) and timing["us"] > 0, "invalid trace timing")
                    require(row["median_us"][timing["label"]] == timing["us"], "single-round median mismatch")
                    expected_logs.append({"kid": row["kid"], "shape": row["shape"],
                        **{key: timing[key] for key in ["round", "label", "us"]}})
                baseline, candidate = row["median_us"]["baseline"], row["median_us"]["candidate"]
                require(row["median_speedup"]["candidate"] == baseline / candidate, "trace ratio mismatch")
                inv = shapes[tuple(row["shape"])]
                require(target["current745_historical_winner_parent"] == inv["parent_id"], "current winner annotation mismatch")
                require(target["purpose"] == "runtime_changed" and not target["formal_dispatch_required"], "support shape dispatch contract mismatch")
                require(record["kid"] == 9054 or row["shape"][2] != 7168, "fixed dispatch incorrectly treated as changed runtime")
                local.append({"parent_id": row["kid"], "shape": row["shape"], "part": command["name"],
                    "target_index": index, "current745_winner_parent": inv["parent_id"],
                    "same_parent_current_winner": inv["parent_id"] == row["kid"],
                    "trace_baseline_us": baseline, "trace_candidate_us": candidate,
                    "trace_candidate_time_change_percent": 100 * (candidate / baseline - 1),
                    "trace_throughput_change_percent": 100 * (baseline / candidate - 1),
                    "timing_interpretation": "single profiler screen; no final Event gain inference"})
            observed_logs = [row for row in json_lines(log_path) if "us" in row and row.get("method") is None]
            require(observed_logs == expected_logs, "complete numerical trace logs differ from JSON")
            clean_rows.extend(local)
            part.update(status="clean_complete_slice", included_in_support_coverage=True,
                claim_epoch={"claim_line": claim_index + 1, "start_line": start_index + 1, "end_line": end_index + 1,
                             "claim_log": log_source,
                             "claim": claim, "start": start, "end": end},
                signed_numerical_calls=len(plan["targets"]) * 2 * 8, log_sha256=sha(log_path),
                rows_numerical_guard_repeatability_passed=True, complete_trace_log_matches=True)
        except (ValueError, KeyError, TypeError, OSError) as exc:
            part.update(status="audit_failed_excluded", included_in_support_coverage=False, reason=str(exc))
    require(len(expected_pairs) == len(set(expected_pairs)) == 646, "planned646 parent-shape domain differs")
    clean_pairs = {(row["parent_id"], *row["shape"]) for row in clean_rows}
    require(len(clean_pairs) == len(clean_rows) and clean_pairs <= set(expected_pairs), "coverage duplicate or outside domain")
    summary = []
    for kid, expected in [(9042, 178), (9053, 178), (9054, 290)]:
        local = [row for row in clean_rows if row["parent_id"] == kid]
        summary.append({"parent_id": kid, "expected_rows": expected, "clean_complete_rows": len(local),
            "all_expected_complete": len(local) == expected,
            "part_states": dict(Counter(part["status"] for part in parts if part["parent_id"] == kid)),
            "signed_numerical_calls": len(local) * 16})
    result = {"status": "complete_support_screen_only" if len(clean_rows) == 646 else "partial_support_pending",
        "analysis_cpu_only": True, "gpu_executed_by_analysis": False, "production_modified": False,
        "source_head": inventory["source_head"], "libraries": libraries,
        "claim_logs": list(claim_logs), "claim_epoch_policy": "Each claim/start/end sequence comes from one complete log; never merge ordered events across wrapper logs.",
        "summary": {"expected_parts": 22, "clean_parts": sum(p["status"] == "clean_complete_slice" for p in parts),
                    "expected_parent_shape_cases": 646, "clean_parent_shape_cases": len(clean_rows),
                    "signed_numerical_calls_passed": len(clean_rows) * 16},
        "parents": summary, "parts": parts, "clean_rows": clean_rows, "archived_attempts": archives,
        "screen_slowdown_candidates": [row for row in clean_rows if row["trace_candidate_time_change_percent"] >= args.screen_slowdown_percent],
        "screen_listing_threshold_percent": args.screen_slowdown_percent,
        "adopted": False,
        "limits": ["Trace1x11 times are descriptive screening only; no final adoption, loss or counter-root-cause claim.",
                   "Only complete clean primary epochs count; partial/contaminated/archive attempts excluded.",
                   "These646 cases cover changed runtime branch support on745 shape universe, not every arbitrary legal tensor.",
                   "Original signed8 checks prove only their buffers; final performance confirmation needs shared-pool Event/all-buffer checks.",
                   "Scoped6winner Event and mechanism commands are separate from22 support slices.",
                   "An isolated official candidate/worktree build remains pending evaluation and is not production adoption."]}
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": result["status"], "summary": result["summary"], "parents": summary}, indent=2))


if __name__ == "__main__":
    main()
