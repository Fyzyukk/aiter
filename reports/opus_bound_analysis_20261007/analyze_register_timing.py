#!/usr/bin/env python3
"""CPU-only summary of the first clean register-reuse timing window."""
from collections import Counter
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]


def read(rel):
    return json.loads((OUT / rel).read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(values):
    med = statistics.median(values)
    return {"n": len(values), "min": min(values), "median": med, "max": max(values),
            "mean": statistics.mean(values), "stdev": statistics.stdev(values),
            "cv_percent": 100 * statistics.stdev(values) / statistics.mean(values),
            "mad_percent": 100 * statistics.median(abs(v - med) for v in values) / med,
            "range_percent_of_median": 100 * (max(values) - min(values)) / med}


def main():
    inventory = read("shape_inventory.json")
    probe = read("shape_inventory_evidence/traits_probe.json")["traits"]
    audit = read("register_reuse/device_audit.json")
    manifest = read("register_reuse/build_manifest.json")
    libraries = {lib["mode"]: lib for lib in audit["libraries"]}
    manifest_libraries = {lib["mode"]: lib for lib in manifest["libraries"]}
    actual_libraries = {side: sha(OUT / "register_reuse" / side / "experiments.so")
                        for side in libraries}
    claims = [json.loads(line) for line in (OUT / "gpu_claim_log.jsonl").read_text().splitlines()]
    spec = importlib.util.spec_from_file_location("register_timing_inventory", OUT / "shape_inventory.py")
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    sys.path.insert(0, str(ROOT / "csrc/opus_gemm"))
    import opus_gemm_common as registry
    from codegen import gen_instances_gfx950 as codegen
    assert "torch" not in sys.modules
    type_to_id = {codegen._bpreshuffle_compact_traits(instance): kid
                  for kid, instance in registry.a8w8_mxscale_gemm_bpreshuffle_kernels_list.items()}

    def formal_dispatch(kid, shape):
        leaves = inventory["parents"][str(kid)]["branches"]
        selected = [leaf for leaf in leaves if all(helpers.cpp_eval(cond, *shape)
                                                   for cond in leaf["conditions"])]
        assert len(selected) == 1
        leaf = selected[0]
        return {"public_id": kid, "actual_id": type_to_id.get(leaf["traits_cpp"], kid),
                "conditions": leaf["conditions"], "canonical_traits_type": probe[leaf["traits_cpp"]]["canonical_type"]}

    rows = []
    for path in sorted((OUT / "results").glob("register_timing_*_kid*.json")):
        data = json.loads(path.read_text())
        assert data["status"] == "passed" and not data.get("contamination")
        assert data["source_head"] == inventory["source_head"]
        assert data["runner_sha256"] == sha(OUT / "experiment_runner.py")
        assert data["plan_sha256"] == sha(OUT / "plans/register_timing.json")
        assert len(data["rows"]) == 1
        row = data["rows"][0]
        kid, shape = row["kid"], row["shape"]
        for side in libraries:
            assert data["libraries"][side]["sha256"] == actual_libraries[side] == libraries[side]["binary_sha256"]
            check = row["correctness"][side]
            assert check["repetitions"] == 8 and check["errRatio"] == 0
            assert all(check[key] for key in ["output_guards", "workspace_guards", "repeatable"])
        event = row["event_confirmation"]
        assert event["status"] == "passed" and event["shared_pool"]
        assert event["rounds"] == 5 and event["iters_per_graph"] == 51
        assert len(event["pool_pointers"]) == event["rotation"]["count"] == 51
        assert len(event["measurements"]) == 10
        rounds = []
        for index in range(5):
            measurements = [m for m in event["measurements"] if m["round"] == index]
            assert len(measurements) == 2
            by_side = {m["label"]: m for m in measurements}
            assert set(by_side) == {"baseline", "candidate"}
            for measurement in measurements:
                assert measurement["iters"] == measurement["rotation_count"] == 51
                assert abs(measurement["us_per_call"] - 1000 * measurement["event_total_ms"] / 51) < 1e-10
                assert len(measurement["all_pool_checks"]) == 51
                for check in measurement["all_pool_checks"]:
                    assert check["errRatio"] == 0
                    assert check["output_repeatable"] and check["output_guards"] and check["workspace_guards"]
                    # Non-split register path has no workspace to compare.
            baseline, candidate = [by_side[side]["us_per_call"] for side in ["baseline", "candidate"]]
            rounds.append({"round": index, "order": measurements[0]["order"],
                           "baseline_us": baseline, "candidate_us": candidate,
                           "speedup_baseline_over_candidate": baseline / candidate,
                           "candidate_time_change_percent": 100 * (candidate / baseline - 1),
                           "saved_us": baseline - candidate})
        by_side = {side: stats([r[side + "_us"] for r in rounds]) for side in ["baseline", "candidate"]}
        for side in by_side:
            assert by_side[side]["median"] == event["median_us"][side]
        speedup = event["median_us"]["baseline"] / event["median_us"]["candidate"]
        paired = stats([r["speedup_baseline_over_candidate"] for r in rounds])
        assert speedup == event["median_speedup"]["candidate"]
        name = path.stem
        starts = [i for i, claim in enumerate(claims) if claim["event"] == "start" and claim["command"]["name"] == name]
        assert starts
        start = starts[-1]
        end = next(i for i in range(start + 1, len(claims))
                   if claims[i]["event"] == "end" and claims[i]["name"] == name)
        interval = claims[start:end + 1]
        assert claims[end]["returncode"] == 0 and not claims[end]["contamination"]
        assert not any(c["event"] == "external_work_started" for c in interval)
        finish_after_monitors = [claim for claim in interval if claim["event"] == "monitor"
                                and claim["time"] > data["finished"] and claim.get("processes")]
        baseline_kernel = next(k for k in libraries["baseline"]["kernels"] if k["variant"] == kid)
        candidate_kernel = next(k for k in libraries["candidate"]["kernels"] if k["variant"] == kid)
        ints = baseline_kernel["template_ints"]
        private_canonical = "opus_gemm_small_register_traits_gfx950<" + ", ".join(
            [*(str(v) for v in ints), "false", "false"]) + ">"
        same_public = formal_dispatch(kid, shape)
        current_winners = [{"parent_id": item["parent_id"], "actual_id": item["actual_configuration_id"],
                            "canonical_traits_type": item["canonical_traits_type"]}
                           for item in inventory["rows"] if [item["M"], item["N"], item["K"]] == shape]
        if kid == 9040:
            decision = "reject_reuse_for_9040; exclude_global_default_change"
        elif kid in [9042, 9053, 9054]:
            decision = "positive_single_window_signal; second_window_and_boundary_regression_required"
        else:
            decision = "no_resolved_gain_in_this_window; do_not_adopt"
        rows.append({"file": str(path.relative_to(OUT)), "file_sha256": sha(path), "kid": kid,
            "shape": shape, "gpu": data["gpu"], "started_utc": datetime.datetime.fromtimestamp(
                data["started"], datetime.timezone.utc).isoformat(),
            "finished_utc": datetime.datetime.fromtimestamp(data["finished"], datetime.timezone.utc).isoformat(),
            "file_status": data["status"], "event_status": event["status"],
            "claim": {"start_line": start + 1, "end_line": end + 1, "returncode": 0,
                      "contamination": False, "external_work_started_count": 0,
                      "process_monitors_after_json_finish": finish_after_monitors},
            "correctness": {"signed_repetitions_per_side": 8, "pool_checks_per_side": 5 * 51,
                            "pool_checks_both_sides": 10 * 51, "all_numeric_guards_repeatability_passed": True},
            "baseline_us": by_side["baseline"]["median"], "candidate_us": by_side["candidate"]["median"],
            "speedup_ratio_of_medians": speedup, "candidate_time_change_percent": 100 * (1 / speedup - 1),
            "event_distribution": by_side, "paired_speedup_distribution": paired,
            "paired_candidate_faster_rounds": sum(r["speedup_baseline_over_candidate"] > 1 for r in rounds),
            "rounds": rounds,
            "order_group_paired_speedup_medians": {" -> ".join(order): statistics.median(
                r["speedup_baseline_over_candidate"] for r in rounds if r["order"] == order)
                for order in [["baseline", "candidate"], ["candidate", "baseline"]]},
            "private_vs_formal": {"private_runtime_symbol": baseline_kernel["name"],
                "private_runtime_canonical_traits": private_canonical,
                "production_dispatch_bypassed": True, "same_public_id_production_dispatch": same_public,
                "same_public_dispatch_matches_private_traits": same_public["canonical_traits_type"] == private_canonical,
                "current745_same_shape_winners": current_winners,
                "current745_private_traits_winner_for_this_shape": any(item["parent_id"] == kid and
                    item["actual_id"] == kid for item in current_winners),
                "transfer_limit": "A/B compares the direct runtime variant only; no alternative current winner was timed and fixedK public dispatch is not benchmarked by this private call."},
            "static_change": {"b8_loads": [baseline_kernel["b8_vmem_count"], candidate_kernel["b8_vmem_count"]],
                              "baseline_resources": baseline_kernel["resources"],
                              "candidate_resources": candidate_kernel["resources"]},
            "trace_screening_us_only": row["median_us"], "decision": decision})
    assert len(rows) == 6
    result = {"status": "cpu_summary_of_six_clean_first_window_events", "gpu_executed_by_this_analysis": False,
        "production_modified": False, "source_head": inventory["source_head"],
        "libraries": {side: {"result_disk_device_audit_sha256": actual_libraries[side],
            "build_manifest_recorded_sha256": manifest_libraries[side]["binary_sha256"],
            "pre_audit_binary_sha256": manifest_libraries[side].get("pre_audit_binary_sha256"),
            "build_manifest_matches_current_binary": manifest_libraries[side]["binary_sha256"] == actual_libraries[side]}
            for side in libraries},
        "methods": {"final_timing": "5 alternating AB/BA HIP-backed Event rounds, graph batch51, same physical input/output pool; ratio of separate medians and paired ratios both shown",
                    "distribution": "sample SD/CV, median absolute deviation and min/max are descriptive only, not confidence intervals",
                    "statistical_limit": "Five sequential rounds in one idle window are correlated; sign consistency is a local check and no p-value/significance claim is made",
                    "trace": "One torch-profiler trace screening round is not used for acceptance",
                    "claim": "Caller logs end returncode0/contaminationfalse for all6; no external_work_started; register5 extra process monitor occurs after JSON finished and cannot establish ownership from simplified record"},
        "rows": rows,
        "decision": {"global_default_change": "rejected:9040 reproducibly regressed all5 rounds despite fewer byte loads/VGPR",
            "9040": "Keep existing false trait; no expanded regression of rejected version needed",
            "9042_9053_9054": "First-window positive signals only; repeat in another clean window, then boundary and actual current-winner checks",
            "9051_9052": "No resolved improvement; mixed paired signs and order sensitivity, keep false pending meaningful new evidence",
            "if_scoped_adoption_later": "Opt in only proven aliases or add an exact trait predicate excluding9040; rebuild/test that candidate, verify official host/device mangled symbols and unchanged fixed9070-9073. Do not copy the one-line global default mutation."},
        "remaining": ["No TA/TCP/TCC/UTCL1/wave counters were collected; no hardware-root-cause inference from timing/resource changes.",
                      "The caller reconciled pre-audit/final SHA fields after objcopy rewrote ELF; current result/disk/device_audit/manifest final identity is checked.",
                      "Use landing_conditions.json and incremental plans for queue, N128 scale group, M tail and formal dispatch coverage."]}
    affected = []
    for item in inventory["rows"]:
        kid = item["actual_configuration_id"]
        if kid not in [9040, 9042, 9051, 9052, 9053, 9054]:
            continue
        # Actual runtime leaves only; fixed9070/9072 already use reuse and
        # N48 must remain false. The runner dispatches this private ID directly.
        shape = [item["M"], item["N"], item["K"]]
        traits = formal_dispatch(item["parent_id"], shape)
        assert traits["actual_id"] == kid and item["fixed_k"] == 0
        affected.append({"kid": kid, "shape": shape, "seed": 301 + len(affected),
            "signed": True, "public_parent_id": item["parent_id"],
            "expected_formal_actual_id": kid,
            "canonical_traits_type": item["canonical_traits_type"],
            "historical_winner_us": item["historical_winner_us"],
            "eligibility": "rejected_9040_do_not_run_for_adoption" if kid == 9040 else (
                "positive_signal_family_current_winner_regression" if kid in [9042, 9053, 9054]
                else "no_resolved_first_window_gain; preserve_false_unless_new_evidence"),
            "private_baseline_matches_current_winner_instructions":
                next(k for k in libraries["baseline"]["kernels"] if k["variant"] == kid)["instruction_sha256"]
                == item["official_instruction_sha256"]})
    assert len(affected) == 41
    assert all(target["private_baseline_matches_current_winner_instructions"] for target in affected)
    plan = {"libraries": read("plans/register_timing.json")["libraries"], "workspace": False,
        "targets": affected,
        "scope": "41 current745 historical winner shapes whose direct runtime branch changes effective B-scale bytecode; CPU-only prepared, not executed",
        "family_counts": dict(sorted(Counter(t["kid"] for t in affected).items())),
        "positive_signal_family_counts": dict(sorted(Counter(t["kid"] for t in affected
            if t["kid"] in [9042, 9053, 9054]).items())),
        "run_guidance": "Do not run all41 indiscriminately. Exclude rejected9040 (11 shapes); prioritize private9042 (1),9053 (2),9054 (3) on their actual winner shapes using filtered plans. No production dispatch change is authorized by this plan."}
    plan_path = OUT / "plans/register_current_affected.json"
    plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n")
    result["current745_affected_plan"] = {"path": str(plan_path.relative_to(OUT)),
        "family_counts": plan["family_counts"], "positive_signal_family_counts": plan["positive_signal_family_counts"],
        "all41_baseline_instruction_hashes_exact_current_winner": True}
    target = OUT / "register_timing_summary.json"
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"summary": [{"kid": row["kid"], "shape": row["shape"],
        "baseline_us": row["baseline_us"], "candidate_us": row["candidate_us"],
        "speedup": row["speedup_ratio_of_medians"], "paired_wins": row["paired_candidate_faster_rounds"],
        "decision": row["decision"]} for row in rows], "global_default": result["decision"]["global_default_change"]},
        indent=2, ensure_ascii=False))
    print(json.dumps(result["current745_affected_plan"], indent=2))


if __name__ == "__main__":
    main()
