#!/usr/bin/env python3
"""CPU audit of finite register closing checks; never executes HIP.

Reuses prior actual winner Events by exact device identity. Audits five new
mechanism cases and six existing support-screen Event followups, with complete
single-log clean epochs and all shared pool checks. Optional646 sweep is not a
completion gate and untested supported fallback shapes are not claimed passed.
"""
from collections import Counter
import json
from pathlib import Path

from prepare_register_minimal_closing import OUT, read, sha, support, verify_event, write


def verify_mechanism(command, queue, logs, libraries):
    argv = command["argv"]
    plan_path = Path(argv[argv.index("--plan") + 1])
    path = Path(argv[argv.index("--output") + 1])
    data, plan = read(path), read(plan_path)
    assert data["status"] == "passed" and data["finished"] and not data.get("contamination")
    assert data["source_head"] == read(OUT / "shape_inventory.json")["source_head"]
    assert data["runner_sha256"] == sha(OUT / "experiment_runner.py")
    assert data["plan"] == plan and data["plan_sha256"] == sha(plan_path)
    assert data["libraries"] == libraries and data["rounds"] == 1 and data["iters"] == 11
    assert not data["profiling_only"] and not data["event_confirmation_requested"]
    assert data["label_filter"] is None and data["target_index"] is None and data["warmup"] == 5
    assert len(data["rows"]) == len(plan["targets"]) == 5
    epochs = []
    for log_source, claims in logs.items():
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
            assert start["fingerprint"] == support.fingerprint(command, queue["env"])
            claim_index, claim = max(((i, c) for i, c in enumerate(claims[:start_index + 1])
                if c.get("event") == "claimed"), key=lambda pair: pair[1]["time"])
            assert data["gpu"]["pci_bdf"] == data["gpu"]["expected_pci_bdf"] == claim["gpu"]["bdf"]
            assert data["gpu"]["HIP_VISIBLE_DEVICES"] == str(claim["gpu"]["hip_index"])
            epochs.append({"claim_log": str(log_source), "claim_line": claim_index + 1,
                           "start_line": start_index + 1, "end_line": end_index + 1})
    assert epochs, "No exact clean single-log mechanism completion"
    expected_logs, rows = [], []
    traits = {9042: (3, 4), 9053: (2, 8), 9054: (2, 4)}
    for index, (row, target) in enumerate(zip(data["rows"], plan["targets"])):
        assert row["target_index"] == index and row["kid"] == target["kid"] and row["shape"] == target["shape"]
        assert target["signed"] and isinstance(target["seed"], int) and row["split"] == 1
        for label in ["baseline", "candidate"]:
            check = row["correctness"][label]
            assert check["repetitions"] == 8 and check["errRatio"] == 0 and check["repeatable"]
            assert check["output_guards"] and check["workspace_guards"]
        assert len(row["timings"]) == 2 and [t["label"] for t in row["timings"]] == ["baseline", "candidate"]
        for timing in row["timings"]:
            assert timing["round"] == 0 and timing["errRatio"] == 0 and timing["output_guards"] and timing["workspace_guards"]
            expected_logs.append({"kid": row["kid"], "shape": row["shape"], **{k: timing[k] for k in ["round", "label", "us"]}})
        prefetch, wave_k = traits[row["kid"]]
        count, extra = divmod(row["shape"][2] // 128, wave_k)
        loops = [count + (wk < extra) for wk in range(wave_k)]
        rows.append({"parent_id": row["kid"], "shape": row["shape"], "prefetch": prefetch,
            "K_waves": wave_k, "wave_tile_counts": loops,
            "full_groups_per_wave": [n // prefetch for n in loops],
            "tail_slots_per_wave": [n % prefetch for n in loops],
            "M33_tail": row["shape"][0] == 33, "N128_scale_groups": row["shape"][1] // 128,
            "signed8_guard_repeatability_passed": True, "timing_interpretation": "trace screening only"})
    actual_logs = [r for r in support.json_lines(Path(command["log"])) if "us" in r and r.get("method") is None]
    assert actual_logs == expected_logs
    return {"status": "five_clean_mechanism_cases_passed", "result": str(path.relative_to(OUT)),
        "result_sha256": sha(path), "log_sha256": sha(Path(command["log"])), "GPU": data["gpu"],
        "claim_epoch": epochs[0], "rows": rows, "signed_numerical_calls": 5 * 16,
        "trace_times_used_for_gain_or_loss": False}


def main():
    queue = read(OUT / "register_minimal_closing_queue.json")
    reuse = read(OUT / "register_reuse_scoped/event_identity_reuse.json")
    assert reuse["status"] == "full_scoped_formal_device_identity_and_prior_events_verified"
    libraries = reuse["scoped_libraries"]
    for v in libraries.values():
        assert v["sha256"] == sha(Path(v["path"]))
    for e in reuse["reusable_events"]:
        assert sha(OUT / e["file"]) == e["sha256"]
    logs = {path: support.json_lines(path) for path in [OUT / "register_scoped_claim_log.jsonl", OUT / "gpu_claim_log.jsonl"] if path.exists()}
    mechanism = verify_mechanism(queue["commands"][0], queue, logs, libraries)
    events = []
    for command in queue["commands"][1:]:
        argv = command["argv"]
        path = Path(argv[argv.index("--output") + 1])
        row = verify_event(path, libraries, [queue], logs)
        row["candidate_time_change_percent"] = 100 * (1 / row["speedup"] - 1)
        row["screen_target"] = read(Path(argv[argv.index("--plan") + 1]))["targets"][int(argv[argv.index("--target-index") + 1])]
        row["current745_same_parent_winner"] = False
        events.append(row)
    assert len(events) == 6
    # Five-percent Event loss is a concrete material-regression screen. A small
    # systematic loss is still reported, not described as proof of zero cost.
    material = [r for r in events if r["candidate_time_change_percent"] >= 5]
    small_negative = [r for r in events if 0 < r["candidate_time_change_percent"] < 5]
    winner = reuse["reusable_events"][:6]
    assert all(r["candidate_faster_rounds"] == 5 for r in winner)
    inventory = read(OUT / "shape_inventory.json")
    formal = read(OUT / "formal_candidate/identity_audit.json")
    changed_symbols = {p["parent_id"]: next(v["baseline"]["name"] for v in p["variants"] if v["changed"])
                       for p in formal["parents"] if p["parent_id"] in [9042, 9053, 9054]}
    affected = [r for r in inventory["rows"] if r["parent_id"] in changed_symbols
                and r["official_kernel_symbol"] == changed_symbols[r["parent_id"]]]
    assert len(affected) == 6
    winner_keys = {(r["parent_id"], *r["shape"]) for r in winner}
    affected_keys = {(r["parent_id"], r["M"], r["N"], r["K"]) for r in affected}
    assert len(affected_keys) == len(winner_keys) == 6 and affected_keys == winner_keys
    assert all(r["fixed_k"] == 0 and r["actual_configuration_id"] == r["parent_id"] for r in affected)
    coverage = {"status": "all_six_effectively_changed_actual_winners_have_clean_Event", "cpu_only": True,
        "inventory_sha256": sha(OUT / "shape_inventory.json"), "formal_audit_sha256": sha(OUT / "formal_candidate/identity_audit.json"),
        "candidate_parent_ids": [9042, 9053, 9054], "definition": "Current745 actual dispatch baseline symbol equals the changed official entry; fixed and other aliases excluded.",
        "affected_actual_winner_count": 6, "Event_covered_count": 6, "remaining_event_targets": [],
        "parent_counts": dict(Counter(r["parent_id"] for r in affected)),
        "rows": [{"parent_id": r["parent_id"], "shape": [r["M"], r["N"], r["K"]],
            "actual_configuration_id": r["actual_configuration_id"], "fixed_k": r["fixed_k"],
            "canonical_traits_type": r["canonical_traits_type"], "baseline_official_symbol": r["official_kernel_symbol"],
            "baseline_official_instruction_sha256": r["official_instruction_sha256"],
            "verified_prior_event": next(e for e in winner if (e["parent_id"], *e["shape"]) == (r["parent_id"], r["M"], r["N"], r["K"]))}
                 for r in affected]}
    write(OUT / "register_reuse_scoped/current_winner_coverage.json", coverage)
    optional = read(OUT / "results/register_scoped_support_analysis.json")
    result = {"status": "finite_register_checks_complete_pending_final_integration" if not material else "material_event_regression_requires_scope_revision",
        "cpu_only": True, "gpu_executed_by_analysis": False, "production_modified": False, "adopted": False,
        "source_head": optional["source_head"], "libraries": libraries,
        "identity_reuse_sha256": sha(OUT / "register_reuse_scoped/event_identity_reuse.json"),
        "optional_support_snapshot": {"sweep_stopped": True, "all646_claimed_passed": False,
            "summary": optional["summary"], "parents": optional["parents"],
            "reference": "results/register_scoped_support_analysis.json"},
        "mechanism": mechanism, "screen_events": events, "prior_actual_winner_events": winner,
        "exact_current_winner_coverage": coverage,
        "prior_support_screen_dedup": read(OUT / "plans/register_scoped_screen_event_followups.json")["deduplicated_prior_event"],
        "additional_prior_mechanism_event": reuse["reusable_events"][-1],
        "summary": {"new_mechanism_cases": 5, "mechanism_signed_numerical_calls": 80,
            "new_screen_event_cases": 6, "new_event_pool_outputs_passed": 3060,
            "prior_actual_winner_events_reused": 6, "prior_winner_pool_outputs_passed": 3060,
            "material_Event_losses_ge5percent": len(material), "small_negative_event_targets": len(small_negative)},
        "material_event_regressions": material, "small_negative_events_reported": small_negative,
        "decision": "No scope rollback required by finite checks. Preserve defaultfalse and opt in only9042/9053/9054; await official API and final integration state. Two lowM/nonwinner controls have consistent sub1percent Event slowdown and must be reported, not called zero regression." if not material else "Restrict or reject changed scope before integration; confirmed material Event losses remain.",
        "limits": ["These are finite mechanism and actual-used winner checks, not exhaustive correctness/performance coverage of arbitrary supported fallback shapes.",
            "Optional646 support sweep stopped at verified210 cases; omitted or polluted cases do not count as passed.",
            "Single trace anomalies are superseded only at matching tested Event shapes; no profiler time is final gain evidence.",
            "LowM9053 K384 round variation is mixed; its positive separate medians do not establish stable optimization.",
            "No new counter/root-cause inference from timing. Exact identity supports reuse, not cross-card extrapolation."]}
    write(OUT / "results/register_minimal_closing_analysis.json", result)
    print(json.dumps({"status": result["status"], "summary": result["summary"],
        "screen_events": [{k: r[k] for k in ["parent_id", "shape", "baseline_us", "candidate_us", "speedup", "candidate_faster_rounds", "candidate_time_change_percent"]} for r in events]}, indent=2))


if __name__ == "__main__":
    main()
