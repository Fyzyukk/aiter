#!/usr/bin/env python3
"""CPU-only evidence summary for nine clean fine_wait Event comparisons."""
from collections import Counter
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


def distribution(values):
    median = statistics.median(values)
    return {"min": min(values), "median": median, "max": max(values),
            "cv_percent": 100 * statistics.stdev(values) / statistics.mean(values),
            "range_percent": 100 * (max(values) - min(values)) / median}


def main():
    inventory = read("shape_inventory.json")
    probe = read("shape_inventory_evidence/traits_probe.json")["traits"]
    manifest = {entry["side"]: entry for entry in read("fine_wait/build_manifest.json")["builds"]}
    audits = {side: read(f"fine_wait/{side}/device_audit.json") for side in ["baseline", "candidate"]}
    kernels = {side: {k["name"]: k for k in entry["kernels"]} for side, entry in audits.items()}
    claims = [json.loads(line) for line in (OUT / "gpu_claim_log.jsonl").read_text().splitlines()]
    spec = importlib.util.spec_from_file_location("fine_timing_inventory", OUT / "shape_inventory.py")
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    sys.path.insert(0, str(ROOT / "csrc/opus_gemm"))
    import opus_gemm_common as registry
    from codegen import gen_instances_gfx950 as codegen
    assert "torch" not in sys.modules
    expression_ids = {codegen._bpreshuffle_compact_traits(instance): kid
                      for kid, instance in registry.a8w8_mxscale_gemm_bpreshuffle_kernels_list.items()}

    def actual_branch(kid, shape):
        if str(kid) in inventory["parents"]:
            selected = [leaf for leaf in inventory["parents"][str(kid)]["branches"]
                        if all(helpers.cpp_eval(condition, *shape) for condition in leaf["conditions"])]
            assert len(selected) == 1
            leaf = selected[0]
            expression = leaf["traits_cpp"]
            actual = expression_ids.get(expression, kid)
            conditions = leaf["conditions"]
        else:
            instance = registry.a8w8_mxscale_gemm_bpreshuffle_kernels_list[kid]
            expression = codegen._bpreshuffle_compact_traits(instance)
            for specialization in instance.bpreshuffle_specializations:
                if shape[2] == specialization[0]:
                    expression = codegen._bpreshuffle_compact_traits(instance, specialization)
                    break
            actual, conditions = kid, ["explicit historical ID"]
        value = probe[expression]
        return actual, conditions, value

    libraries = {}
    for side in ["baseline", "candidate"]:
        binary = OUT / "fine_wait" / side / "experiments.so"
        checksum = sha(binary)
        assert checksum == audits[side]["binary_sha256"] == manifest[side]["binary_sha256"]
        libraries[side] = {"sha256": checksum, "disk_device_audit_manifest_equal": True}

    rows = []
    for path in sorted((OUT / "results").glob("fine_timing_*_kid*.json")):
        data = json.loads(path.read_text())
        assert data["status"] == "passed" and not data.get("contamination")
        assert data["source_head"] == inventory["source_head"]
        assert data["runner_sha256"] == sha(OUT / "experiment_runner.py")
        assert data["plan_sha256"] == sha(OUT / "plans/fine_timing.json")
        assert len(data["rows"]) == 1
        row = data["rows"][0]
        kid, shape = row["kid"], row["shape"]
        actual, conditions, traits = actual_branch(kid, shape)
        assert traits["SPLIT_K"] == row["split"]
        expression = "gemm_a8w8_mxfp8_scale_small_lds_kernel<" + traits["canonical_type"] + ">"
        # Existing official inventory defines a symbol for every emitted type.
        official = read("shape_inventory_evidence/official_resources.json")
        matched = [entry for parent in official.values() for entry in parent["variants"]
                   if expression.replace(" ", "") in entry["demangled"].replace(" ", "")]
        assert matched
        symbol = matched[0]["name"]
        assert all(kernels["baseline"][symbol]["instruction_sha256"] == k["instruction_sha256"] for k in matched)
        event = row["event_confirmation"]
        assert event["status"] == "passed" and event["shared_pool"]
        assert event["rounds"] == 5 and event["iters_per_graph"] == 51
        assert len(event["measurements"]) == 10 and len(event["pool_pointers"]) == 51
        for side in ["baseline", "candidate"]:
            assert data["libraries"][side]["sha256"] == libraries[side]["sha256"]
            check = row["correctness"][side]
            assert check["repetitions"] == 8 and check["errRatio"] == 0
            assert all(check[key] for key in ["output_guards", "workspace_guards", "repeatable"])
        rounds = []
        for round_index in range(5):
            measurements = [m for m in event["measurements"] if m["round"] == round_index]
            assert len(measurements) == 2
            by_side = {m["label"]: m for m in measurements}
            assert set(by_side) == {"baseline", "candidate"}
            for measurement in measurements:
                assert measurement["rotation_count"] == measurement["iters"] == 51
                assert abs(measurement["us_per_call"] - 1000 * measurement["event_total_ms"] / 51) < 1e-10
                assert len(measurement["all_pool_checks"]) == 51
                for check in measurement["all_pool_checks"]:
                    assert check["errRatio"] == 0
                    assert check["output_guards"] and check["workspace_guards"] and check["output_repeatable"]
                    if traits["SPLIT_K"] > 1:
                        assert check["workspace_repeatable"]
            baseline, candidate = [by_side[side]["us_per_call"] for side in ["baseline", "candidate"]]
            rounds.append({"round": round_index, "order": measurements[0]["order"],
                           "baseline_us": baseline, "candidate_us": candidate, "speedup": baseline / candidate})
        side_distribution = {side: distribution([r[side + "_us"] for r in rounds])
                             for side in ["baseline", "candidate"]}
        for side in side_distribution:
            assert side_distribution[side]["median"] == event["median_us"][side]
        ratio = event["median_us"]["baseline"] / event["median_us"]["candidate"]
        assert ratio == event["median_speedup"]["candidate"]
        name = path.stem
        start = next(i for i in range(len(claims) - 1, -1, -1) if claims[i]["event"] == "start"
                     and claims[i]["command"]["name"] == name)
        end = next(i for i in range(start + 1, len(claims)) if claims[i]["event"] == "end" and claims[i]["name"] == name)
        assert claims[end]["returncode"] == 0 and not claims[end]["contamination"]
        assert not any(c["event"] == "external_work_started" for c in claims[start:end + 1])
        current = [{"parent_id": item["parent_id"], "actual_id": item["actual_configuration_id"],
                    "canonical_traits_type": item["canonical_traits_type"]}
                   for item in inventory["rows"] if [item["M"], item["N"], item["K"]] == shape]
        total = shape[2] // 128
        per, extra = divmod(total, traits["SPLIT_K"])
        rows.append({"file": str(path.relative_to(OUT)), "sha256": sha(path), "kid": kid,
            "shape": shape, "actual_id": actual, "conditions": conditions,
            "canonical_traits_type": traits["canonical_type"], "BM": traits["B_M"], "BN": traits["B_N"],
            "waves": traits["NUM_WAVES"], "split_k": traits["SPLIT_K"], "stages": traits["NUM_STAGES"],
            "cluster": traits["CLUSTER"], "fixed_k": traits["FIXED_K"],
            "reduce_vec": traits["REDUCE_VEC"], "reduce_block": traits["REDUCE_BLOCK"], "cache": traits["STORE_CACHE"],
            "split_loops": [per + (s < extra) for s in range(traits["SPLIT_K"])],
            "file_and_event_passed": True, "claim_end_clean": True,
            "GPU": data["gpu"], "claim_start_line": start + 1, "claim_end_line": end + 1,
            "baseline_instruction_matches_official_current": True,
            "correctness": {"signed_repetitions_per_side": 8, "pool_output_checks": 510,
                "pool_partial_checks": 510 if traits["SPLIT_K"] > 1 else 0,
                "all_numeric_guard_and_repeatability_checks_passed": True},
            "timing_scope": "complete producer+reducer including inter-launch device gaps" if traits["SPLIT_K"] > 1 else "complete direct producer call",
            "baseline_us": event["median_us"]["baseline"], "candidate_us": event["median_us"]["candidate"],
            "speedup": ratio, "candidate_time_change_percent": 100 * (1 / ratio - 1),
            "candidate_faster_rounds": sum(r["speedup"] > 1 for r in rounds),
            "paired_speedup_distribution": distribution([r["speedup"] for r in rounds]),
            "side_distributions": side_distribution, "rounds": rounds,
            "current745_same_shape_winners": current,
            "tested_branch_is_current745_winner": any(item["canonical_traits_type"] == traits["canonical_type"] for item in current),
            "decision": "reject_this_full_fine_wait_mutation; no production adoption"})
    assert len(rows) == 9
    result = {"status": "cpu_verified_nine_clean_fine_wait_events_global_rejected",
        "gpu_executed_by_this_analysis": False, "production_modified": False,
        "source_head": inventory["source_head"], "libraries": libraries, "rows": rows,
        "summary": {"targets": 9, "all5rounds_regression_targets": sum(r["candidate_faster_rounds"] == 0 for r in rows),
                    "median_regression_targets": sum(r["speedup"] < 1 for r in rows),
                    "current745_winner_targets": sum(r["tested_branch_is_current745_winner"] for r in rows),
                    "pool_output_checks_passed": sum(r["correctness"]["pool_output_checks"] for r in rows),
                    "pool_partial_checks_passed": sum(r["correctness"]["pool_partial_checks"] for r in rows)},
        "decision": "Do not adopt the original fine_wait helper/three-site global mutation: five target regressions are consistent across all rounds, the remaining near-zero/mixed signals do not establish a useful gain.",
        "limits": ["The source-level ordering proof and these finite GPU checks do not prove every boundary or arbitrary VMEM mix.",
                   "No TA/TCP/TCC/UTCL1/wave counters collected; timing does not prove that branch/SGPR cost caused the regression.",
                   "The current source uses a conservative valid wait; rejected optimization does not establish a correctness defect.",
                   "Five rounds in one window are descriptive correlated observations, not an independent statistical experiment.",
                   "Rejecting this version needs no extra GPU boundary/regression expansion; future distinct scoped designs would require fresh evidence."]}
    (OUT / "fine_timing_summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"summary": result["summary"], "rows": [{key: row[key] for key in
        ["kid", "shape", "actual_id", "split_k", "baseline_us", "candidate_us", "speedup",
         "candidate_faster_rounds", "tested_branch_is_current745_winner"]} for row in rows]}, indent=2))


if __name__ == "__main__":
    main()
