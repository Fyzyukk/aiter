#!/usr/bin/env python3
"""CPU-only preparation of the remaining register mechanism/Event checks.

Creates new plan/queue files only. Does not edit a running queue, load HIP,
execute a GPU, adopt code, or require optional complete fallback sweeps.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


identity = module("register_closing_identity", OUT / "audit_formal_candidate.py")
support = module("register_closing_support", OUT / "analyze_register_scoped_support.py")


def verify_event(path, libraries, queues, claim_logs):
    data = read(path)
    assert data["status"] == "passed" and data.get("finished") and not data.get("contamination")
    assert len(data["rows"]) == 1 and data["event_confirmation_requested"]
    assert data["source_head"] == read(OUT / "shape_inventory.json")["source_head"]
    assert data["runner_sha256"] == sha(OUT / "experiment_runner.py")
    row = data["rows"][0]
    assert row["split"] == 1 and data["rounds"] == 1 and data["iters"] == 51
    assert data["libraries"] == libraries
    candidates = []
    for queue in queues:
        for command in queue["commands"]:
            if command["name"] != path.stem:
                continue
            argv = command["argv"]
            plan_path = Path(argv[argv.index("--plan") + 1])
            assert data["plan"] == read(plan_path) and data["plan_sha256"] == sha(plan_path)
            index = int(argv[argv.index("--target-index") + 1])
            target = data["plan"]["targets"][index]
            assert row["target_index"] == data["target_index"] == index
            assert target["kid"] == row["kid"] and target["shape"] == row["shape"] and target["signed"]
            for source, claims in claim_logs.items():
                for start_index, start in enumerate(claims):
                    if start.get("event") != "start" or start.get("command") != command:
                        continue
                    ends = [(i, c) for i, c in enumerate(claims[start_index + 1:], start_index + 1)
                            if c.get("event") == "end" and c.get("name") == command["name"]]
                    if not ends:
                        continue
                    end_index, end = ends[0]
                    if not (start["time"] <= data["started"] <= data["finished"] <= end["time"]):
                        continue
                    assert end["returncode"] == 0 and end["contamination"] is False
                    assert not any(c.get("event") == "external_work_started" for c in claims[start_index:end_index + 1])
                    assert start["fingerprint"] == support.fingerprint(command, queue.get("env", {}))
                    claim_index, claim = max(((i, c) for i, c in enumerate(claims[:start_index + 1])
                        if c.get("event") == "claimed"), key=lambda pair: pair[1]["time"])
                    assert data["gpu"]["pci_bdf"] == data["gpu"]["expected_pci_bdf"] == claim["gpu"]["bdf"]
                    assert data["gpu"]["HIP_VISIBLE_DEVICES"] == str(claim["gpu"]["hip_index"])
                    candidates.append({"command": command, "log": Path(command["log"]),
                        "claim_epoch": {"claim_log": str(source), "claim_line": claim_index + 1,
                                        "start_line": start_index + 1, "end_line": end_index + 1}})
    assert candidates, "No exact clean single-log claim epoch for " + str(path)
    evidence = candidates[0]
    event = row["event_confirmation"]
    assert event["status"] == "passed" and event["shared_pool"] and event["rounds"] == 5
    assert event["iters_per_graph"] == event["rotation"]["count"] == len(event["pool_pointers"]) == 51
    assert len(event["measurements"]) == 10
    for label in ["baseline", "candidate"]:
        check = row["correctness"][label]
        assert check["repetitions"] == 8 and check["errRatio"] == 0 and check["repeatable"]
        assert check["output_guards"] and check["workspace_guards"]
    expected_logs = []
    for timing in row["timings"]:
        assert timing["errRatio"] == 0 and timing["output_guards"] and timing["workspace_guards"]
        expected_logs.append({"kid": row["kid"], "shape": row["shape"],
            **{k: timing[k] for k in ["round", "label", "us"]}})
    faster = 0
    for rnd in range(5):
        measurements = [m for m in event["measurements"] if m["round"] == rnd]
        order = ["baseline", "candidate"] if rnd % 2 == 0 else ["candidate", "baseline"]
        assert [m["label"] for m in measurements] == order
        by = {m["label"]: m for m in measurements}
        faster += by["candidate"]["us_per_call"] < by["baseline"]["us_per_call"]
        for m in measurements:
            assert m["order"] == order and m["iters"] == m["rotation_count"] == len(m["all_pool_checks"]) == 51
            assert abs(m["us_per_call"] - 1000 * m["event_total_ms"] / 51) < 1e-10
            assert [c["pool_index"] for c in m["all_pool_checks"]] == list(range(51))
            assert all(c["errRatio"] == 0 and c["output_repeatable"] and c["output_guards"]
                       and c["workspace_guards"] for c in m["all_pool_checks"])
            expected_logs.append({"method": "graph_event_confirmation", "kid": row["kid"], "shape": row["shape"],
                **{k: m[k] for k in ["round", "label", "event_total_ms", "iters", "us_per_call", "rotation_count"]}})
    actual_logs = [r for r in support.json_lines(evidence["log"]) if "us" in r or r.get("method") == "graph_event_confirmation"]
    assert actual_logs == expected_logs
    return {"file": str(path.relative_to(OUT)), "sha256": sha(path), "parent_id": row["kid"],
        "shape": row["shape"], "baseline_us": event["median_us"]["baseline"],
        "candidate_us": event["median_us"]["candidate"], "speedup": event["median_speedup"]["candidate"],
        "candidate_faster_rounds": faster, "pool_outputs_passed": 510,
        "all_signed8_guard_repeatability_and_event_pool_checks_passed": True,
        "claim_epoch": evidence["claim_epoch"], "log_sha256": sha(evidence["log"]),
        "whole_library_hash_diff_deduplicated_by_full_device_identity": True}


def main():
    formal = read(OUT / "formal_candidate/identity_audit.json")
    assert not formal["failures"] and formal["cpu_only"] and not formal["production_checkout_modified"]
    full = {side: identity.rows(OUT / "register_reuse" / side / "experiments.so") for side in ["baseline", "candidate"]}
    scoped = {side: identity.rows(OUT / "register_reuse_scoped" / side / "experiments.so") for side in ["baseline", "candidate"]}
    comparisons = []
    for parent in formal["parents"]:
        if parent["parent_id"] not in [9042, 9053, 9054]:
            continue
        changed = next(v for v in parent["variants"] if v["changed"])
        for side in ["baseline", "candidate"]:
            name = changed[side]["name"]
            a = next(r for r in full[side] if r["name"] == name)
            b = next(r for r in scoped[side] if r["name"] == name)
            checks = {"full_to_scoped": identity.identity(a, b), "scoped_to_formal": identity.identity(b, changed[side])}
            assert all(identity.all_matched(c) for c in checks.values())
            comparisons.append({"parent_id": parent["parent_id"], "side": side, "checks": checks,
                                "original_full": a, "scoped": b, "formal": changed[side]})
    original_libraries = {side: {"path": str(OUT / "register_reuse" / side / "experiments.so"),
        "sha256": sha(OUT / "register_reuse" / side / "experiments.so")} for side in ["baseline", "candidate"]}
    queues = [read(OUT / "confirmation_idle_queue.json"), read(OUT / "regression_and_profile_idle_queue.json"),
              read(OUT / "focused_idle_queue.json")]
    claim_logs = {OUT / "gpu_claim_log.jsonl": support.json_lines(OUT / "gpu_claim_log.jsonl")}
    events = [verify_event(OUT / "results" / f"register_positive_winner_{i}.json", original_libraries, queues, claim_logs) for i in range(6)]
    events.append(verify_event(OUT / "results/register_timing_4_kid9053.json", original_libraries, queues, claim_logs))
    events.append(verify_event(OUT / "results/register_timing_5_kid9054.json", original_libraries, queues, claim_logs))
    write(OUT / "register_reuse_scoped/event_identity_reuse.json", {
        "status": "full_scoped_formal_device_identity_and_prior_events_verified", "cpu_only": True,
        "gpu_executed": False, "production_modified": False, "formal_audit_sha256": sha(OUT / "formal_candidate/identity_audit.json"),
        "original_libraries": original_libraries, "scoped_libraries": {side: {"path": str(OUT / "register_reuse_scoped" / side / "experiments.so"),
            "sha256": sha(OUT / "register_reuse_scoped" / side / "experiments.so")} for side in ["baseline", "candidate"]},
        "comparisons": comparisons, "reusable_events": events,
        "limits": ["Exact instruction, complete metadata and normalized descriptor matching permits reuse for the same shape and call contract.",
                   "Existing measured timings remain from their own clean physical card/window; device identity does not establish new cross-card gains.",
                   "Six positive actual winner Events are complete; arbitrary fallback support is not claimed fully tested."]})
    screens = read(OUT / "results/register_scoped_support_analysis.json")["screen_slowdown_candidates"]
    assert len(screens) == 7, "Support snapshot changed; review before preparing a new queue"
    deduplicated, targets = [], []
    for screen in screens:
        prior = next((e for e in events if e["parent_id"] == screen["parent_id"] and e["shape"] == screen["shape"]), None)
        if prior:
            deduplicated.append({"screen": screen, "prior_event": prior,
                "decision": "Existing clean 5-round all-buffer Event overrides the single trace screen; no duplicate timing required."})
            continue
        targets.append({"kid": screen["parent_id"], "shape": screen["shape"], "seed": 1201 + len(targets),
            "signed": True, "purpose": "verify_material_single_trace_slowdown", "screen": screen})
    assert len(targets) == 6 and len(deduplicated) == 1
    plan_path = OUT / "plans/register_scoped_screen_event_followups.json"
    libraries = {side: str(OUT / "register_reuse_scoped" / side / "experiments.so") for side in ["baseline", "candidate"]}
    write(plan_path, {"libraries": libraries, "workspace": False, "targets": targets,
        "purpose": "Finite Event confirmation of existing clean support screen anomalies; optional full646 sweep stopped.",
        "prepared_cpu_only": True, "deduplicated_prior_event": deduplicated})
    prior_queue = read(OUT / "register_scoped_idle_queue.json")
    commands = [next(c for c in prior_queue["commands"] if c["name"] == "register_scoped_mechanism")]
    for index in range(len(targets)):
        name = f"register_scoped_screen_event_{index}"
        commands.append({"name": name, "argv": ["/opt/venv/bin/python3", str(OUT / "experiment_runner.py"),
            "--plan", str(plan_path), "--target-index", str(index), "--output", str(OUT / "results" / (name + ".json")),
            "--rounds", "1", "--iters", "51", "--repetitions", "8", "--event-confirm"],
            "log": str(OUT / "results" / (name + ".log"))})
    queue_path = OUT / "register_minimal_closing_queue.json"
    write(queue_path, {"env": prior_queue["env"], "independent_commands_defer_interrupted": True,
        "commands": commands, "existing_six_positive_events_reused": True,
        "optional_full646_support_sweep_stopped": True})
    assert "torch" not in sys.modules
    print(json.dumps({"queue": str(queue_path), "mechanism_cases": 5, "new_event_cases": len(targets),
        "prior_positive_events_reused": 6, "screen_events_deduplicated": 1, "gpu_executed": False}, indent=2))


if __name__ == "__main__":
    main()
