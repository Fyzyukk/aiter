#!/usr/bin/env python3
"""CPU-only evidence for register winner confirmation and clean N64 rejection."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import sys

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(rel):
    return json.loads((OUT / rel).read_text())


def dist(values):
    median = statistics.median(values)
    return {"min": min(values), "median": median, "max": max(values),
            "range_percent": 100 * (max(values) - min(values)) / median,
            "cv_percent": 100 * statistics.stdev(values) / statistics.mean(values)}


def verify(path, claims):
    data = json.loads(path.read_text())
    assert data["status"] == "passed" and not data.get("contamination")
    assert len(data["rows"]) == 1
    row = data["rows"][0]
    event = row["event_confirmation"]
    assert event["status"] == "passed" and event["shared_pool"]
    assert event["rounds"] == 5 and len(event["measurements"]) == 10
    assert event["iters_per_graph"] == event["rotation"]["count"] == len(event["pool_pointers"]) == 51
    for label in ["baseline", "candidate"]:
        check = row["correctness"][label]
        assert check["repetitions"] == 8 and check["errRatio"] == 0
        assert check["repeatable"] and check["output_guards"] and check["workspace_guards"]
        assert data["libraries"][label]["sha256"] == sha(Path(data["libraries"][label]["path"]))
    rounds = []
    for i in range(5):
        measurements = [m for m in event["measurements"] if m["round"] == i]
        assert len(measurements) == 2
        by = {m["label"]: m for m in measurements}
        for m in measurements:
            assert len(m["all_pool_checks"]) == 51 and m["iters"] == 51
            assert abs(m["us_per_call"] - 1000 * m["event_total_ms"] / 51) < 1e-10
            for check in m["all_pool_checks"]:
                assert check["errRatio"] == 0 and check["output_repeatable"]
                assert check["output_guards"] and check["workspace_guards"]
                if row["split"] > 1:
                    assert check["workspace_repeatable"]
        baseline, candidate = [by[label]["us_per_call"] for label in ["baseline", "candidate"]]
        rounds.append({"round": i, "order": measurements[0]["order"], "baseline_us": baseline,
                       "candidate_us": candidate, "speedup": baseline / candidate})
    name = path.stem
    start = next(i for i in range(len(claims) - 1, -1, -1) if claims[i]["event"] == "start"
                 and claims[i]["command"]["name"] == name)
    end = next(i for i in range(start + 1, len(claims)) if claims[i]["event"] == "end" and claims[i]["name"] == name)
    assert claims[end]["returncode"] == 0 and not claims[end]["contamination"]
    assert not any(c["event"] == "external_work_started" for c in claims[start:end + 1])
    ratio = event["median_us"]["baseline"] / event["median_us"]["candidate"]
    assert ratio == event["median_speedup"]["candidate"]
    result = {"file": str(path.relative_to(OUT)), "sha256": sha(path), "kid": row["kid"], "shape": row["shape"],
        "source_head": data["source_head"], "GPU": data["gpu"], "split": row["split"], "libraries": data["libraries"],
        "baseline_us": event["median_us"]["baseline"], "candidate_us": event["median_us"]["candidate"],
        "speedup": ratio, "candidate_time_change_percent": 100 * (1 / ratio - 1),
        "candidate_faster_rounds": sum(r["speedup"] > 1 for r in rounds),
        "rounds": rounds, "paired_speedup": dist([r["speedup"] for r in rounds]),
        "side_distributions": {label: dist([r[label + "_us"] for r in rounds]) for label in ["baseline", "candidate"]},
        "output_pool_checks": 510, "partial_pool_checks": 510 if row["split"] > 1 else 0,
        "all_correctness_guard_repeatability_passed": True,
        "claim": {"start_line": start + 1, "end_line": end + 1, "clean": True},
        "timing_scope": "complete producer+reducer" if row["split"] > 1 else "complete direct runtime call"}
    return result


def main():
    claims = [json.loads(line) for line in (OUT / "gpu_claim_log.jsonl").read_text().splitlines()]
    inventory = load("shape_inventory.json")
    positive = []
    for path in sorted((OUT / "results").glob("register_positive_winner_*.json")):
        row = verify(path, claims)
        winner = [item for item in inventory["rows"] if [item["M"], item["N"], item["K"]] == row["shape"]]
        assert len(winner) == 1 and winner[0]["parent_id"] == winner[0]["actual_configuration_id"] == row["kid"]
        row["current745_winner_exact"] = True
        row["current_official_baseline_instruction_sha256"] = winner[0]["official_instruction_sha256"]
        positive.append(row)
    assert len(positive) == 6
    positive_result = {"status": "six_clean_actual_winners_positive_but_full_support_regression_pending",
        "gpu_executed_by_this_analysis": False, "production_modified": False, "rows": positive,
        "summary": {"targets": 6, "all5rounds_positive_targets": 6, "pool_outputs_passed": 3060,
                    "family_counts": dict(sorted(Counter(r["kid"] for r in positive).items()))},
        "decision": "Evidence supports a scoped opt-in experiment for runtime9042/9053/9054. Keep all other aliases false; fixed9070-9073 retain existing explicit flags. The six winner shapes do not cover all supported shapes.",
        "limits": ["LongK9053 improvement0.445% is locally consistent but small; do not present it as a broad meaningful gain.",
                   "No counter-based cause established. Correlated five-round samples are descriptive.",
                   "The original globaldefault and fine_wait remain rejected."]}
    (OUT / "register_positive_winner_summary.json").write_text(json.dumps(positive_result, indent=2, ensure_ascii=False) + "\n")
    n64 = [verify(path, claims) for path in sorted((OUT / "results").glob("n64_clean_target_*.json"))]
    assert len(n64) == 2
    assert all(row["candidate_faster_rounds"] == 0 for row in n64)
    n64_result = {"status": "two_clean_N64_targets_rejected", "gpu_executed_by_this_analysis": False,
        "production_modified": False, "rows": n64,
        "summary": {"targets": 2, "pool_outputs_passed": 1020, "pool_partials_passed": 1020},
        "decision": "Reject BN128->BN64 replacement for M144/M160,N7168,K16384 M80split2; keep existing BN128.",
        "limits": ["The two earlier interrupted trials remain excluded from all timing/adoption conclusions.",
                   "No counters collected; increased A traffic, dispatch or cache pressure are hypotheses, not proven regression causes.",
                   "Resource reductions alone did not produce throughput gain; no expanded regression of this rejected candidate is required."]}
    (OUT / "n64_clean_summary.json").write_text(json.dumps(n64_result, indent=2, ensure_ascii=False) + "\n")
    sys.path.insert(0, str(ROOT / "csrc/opus_gemm"))
    import opus_gemm_common as registry
    assert "torch" not in sys.modules
    # Public dispatch controls are included; private direct variants do not
    # stand in for unchanged K7168 branches. A future official runner must
    # retain the existing exact dispatch rather than force a private runtime.
    plan_library = load("plans/register_timing.json")["libraries"]
    scoped_audit_path = OUT / "register_reuse_scoped/device_audit.json"
    scoped_ready = scoped_audit_path.exists()
    if scoped_ready:
        scoped = json.loads(scoped_audit_path.read_text())
        assert scoped["status"] == "scoped_aliases_match_tested_machine_code"
        assert all(case["matches_expected"] for case in scoped["comparisons_to_existing_tested_bundles"])
        plan_library = {side: str(OUT / "register_reuse_scoped" / side / "experiments.so")
                        for side in ["baseline", "candidate"]}
    shards = []
    support = {}
    for kid in [9042, 9053, 9054]:
        instance = registry.a8w8_mxscale_gemm_bpreshuffle_kernels_list[kid]
        rows = [item for item in inventory["rows"] if registry.a8w8_mxscale_bpreshuffle_supports_shape(
            instance, item["M"], item["N"], item["K"])]
        changed = [r for r in rows if kid == 9054 or r["K"] != 7168]
        fixed = [r for r in rows if kid != 9054 and r["K"] == 7168]
        support[kid] = {"all_supported_current745_shapes": len(rows), "changed_runtime_shapes": len(changed),
                        "unchanged_fixed_controls": len(fixed)}
        for kind, subset in [("runtime_changed", changed)]:
            for start in range(0, len(subset), 32):
                targets = [{"kid": kid, "shape": [r["M"], r["N"], r["K"]],
                    "seed": 701 + start + i, "signed": True, "formal_dispatch_required": kind == "fixed_control",
                    "current745_historical_winner_parent": r["parent_id"],
                    "purpose": kind} for i, r in enumerate(subset[start:start + 32])]
                name = f"plans/register_scoped_{kid}_{kind}_{start // 32:02d}.json"
                plan = {"libraries": plan_library, "workspace": False, "targets": targets,
                    "prepared_only": True, "runner_contract": "Scoped8-entry direct runtime candidate matches tested ISA. Record librarySHA and use actual public runtime branches; fixed controls are separate.",
                    "production_dispatch_controls_cannot_run_on_direct_private_launch": kind == "fixed_control"}
                (OUT / name).write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n")
                shards.append({"path": name, "kid": kid, "kind": kind, "targets": len(targets),
                    "direct_private_runtime_can_cover": kind == "runtime_changed",
                    "needs_scoped_or_official_library_binding": True})
    assert [support[k]["all_supported_current745_shapes"] for k in support] == [290, 290, 290]
    assert sum(v["changed_runtime_shapes"] for v in support.values()) == 646
    assert sum(v["unchanged_fixed_controls"] for v in support.values()) == 224
    fixed_controls = [
        {"kid": 9042, "shape": [256, 768, 7168], "expected_actual_id": 9071, "purpose": "N48M16 grid256 equality"},
        {"kid": 9042, "shape": [257, 768, 7168], "expected_actual_id": 9073, "purpose": "N48M32 after grid threshold, M tail"},
        {"kid": 9053, "shape": [33, 384, 7168], "expected_actual_id": 9072, "purpose": "fixedReuseTrue output4 M tail"},
    ]
    support_result = {"status": "CPU_prepared_scoped_support_regression_not_executed", "support": support,
        "shards": shards, "counts": {"parent_shape_supported_domain": 870, "runtime_changed": 646,
            "fixed_unchanged_domain_checked_by_hash": 224, "fixed_optional_gpu_smokes": 3, "runtime_shards": len(shards)},
        "scoped_private_library_ready": scoped_ready,
        "scoped_device_audit": "register_reuse_scoped/device_audit.json" if scoped_ready else None,
        "fixed_controls": fixed_controls,
        "execution_contract": ["Execute only after scoped candidate ISA identity checked and libraries rebound; fixed control cases require production-equivalent dispatch.",
            "Existing six actual winners and prior valid boundaries may be deducted by matching shape/traits/binary/runner instead of redoing them.",
            "Correctness/guards across support are required; Event performance regression gates should be sufficient to detect material losses on altered nonwinner supported shapes.",
            "Unchanged224 fixed branches are covered by exact resource/hash and dispatch audit; only3 optional GPU boundary smokes listed, not224 repeat tests. Do not force runtime K7168 and call it production coverage."],
        "gpu_executed": False, "production_modified": False}
    (OUT / "register_scoped_support_regression.json").write_text(json.dumps(support_result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"positive": positive_result["summary"], "N64": n64_result["summary"],
                      "support": support, "shards": len(shards)}, indent=2))


if __name__ == "__main__":
    main()
