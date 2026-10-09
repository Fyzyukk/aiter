#!/usr/bin/env python3
"""CPU review of independent private A/B passes; never synthesize dispatch pairs."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import statistics

OUT = Path(__file__).resolve().parent
ROOT = OUT / "ab_profiles"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def claim_for(name, app, claims):
    starts = [(i, row) for i, row in enumerate(claims)
              if row.get("event") == "start" and row.get("command", {}).get("name") == name]
    clean, excluded = None, []
    for i, start in reversed(starts):
        next_start = next((j for j in range(i + 1, len(claims))
                           if claims[j].get("event") == "start"
                           and claims[j].get("command", {}).get("name") == name), len(claims))
        end = next((row for row in claims[i + 1:next_start]
                    if row.get("event") == "end" and row.get("name") == name), None)
        if end and (end["returncode"] != 0 or end.get("contamination")):
            excluded.append({"name": name, "start": start["time"], "end": end["time"],
                             "returncode": end["returncode"], "contamination": end.get("contamination", False)})
        if (clean is None and end and end["returncode"] == 0 and not end.get("contamination")
                and start["time"] <= app["started"] <= app["finished"] <= end["time"]):
            clean = {"start": start, "end": end}
    assert clean is not None, f"No clean claim enclosing application: {name}"
    assert not any(row.get("event") == "external_work_started"
                   and clean["start"]["time"] <= row["time"] <= clean["end"]["time"] for row in claims), name
    return clean, excluded


def main():
    queue_path = OUT / "ab_mechanism_profile_queue.json"
    queue = json.loads(queue_path.read_text())
    spec = importlib.util.spec_from_file_location("ab_mechanism_counter_parser", OUT / "parse_counters.py")
    parser = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parser)
    claims = [json.loads(line) for line in (OUT / "gpu_claim_log.jsonl").read_text().splitlines()]
    report = {
        "status": "cpu_review_no_gpu_access", "queue_sha256": sha(queue_path),
        "collections": [], "comparisons": [], "pending": [], "excluded": [],
        "excluded_claim_attempts": [],
        "comparison_policy": "Only baseline/candidate medians of the same family and PMC group are compared. Each source/dispatch set remains independent; no paired dispatches or cross-group wave decomposition is created.",
        "limits": [
            "Profiler duration is instrumented; a duration delta is not Event-confirmed optimization speedup.",
            "Separate processes preserve input values/seeds but do not share physical addresses or cache state. One A/B pass pair is not a repeated counter experiment.",
            "Short-kernel GRBM clock/window remains unresolved; no absolute MFMA utilization, dynamic occupancy, fixed-factor correction, or cycle-to-time conversion is justified.",
            "SQ wave ratios use matched cumulative wave quad-cycle scope; ANY does not isolate VMEM/VALU/LDS and cannot be subtracted from wall time.",
            "SQ address FIFO is unwindowed and lacks matching SQ_BUSY scope. Its raw delta is only a clue; the derived percentage is not a time fraction.",
            "TCP gate and probe stalls do not support strict windowing; raw/gate changes do not independently establish root cause or saved elapsed cycles.",
            "Scalar counters discard CU/channel/set instance distributions; no hotspot can be confirmed.",
            "Physical DRAM bytes may differ with cache/transaction state; do not infer expected DRAM savings from fewer static scale instructions.",
            "Profiling mode performs no reference/guard checking. Clean same-source correctness and final Event evidence remain separate prerequisites.",
        ],
    }
    # Frozen source hashes are checked read-only. This is separate from the
    # application-recorded measurement-time hashes, which are authoritative.
    report["current_source_hash_matches_queue"] = {
        path: Path(path).exists() and sha(Path(path)) == digest
        for path, digest in queue["source_file_sha256"].items()
    }
    for command in queue["commands"]:
        name, group, label = command["name"], command["group"], command["label"]
        expected = command["expected_target"]
        folder = ROOT / name
        app_path = folder / "application.json"
        if not app_path.exists():
            report["pending"].append(name)
            continue
        app = json.loads(app_path.read_text())
        if app.get("status") != "passed" or app.get("contamination"):
            report["excluded"].append({"name": name, "status": app.get("status"),
                                       "contamination": app.get("contamination"),
                                       "reason": "Entire incomplete/contaminated collection is excluded."})
            continue
        claim, bad_attempts = claim_for(name, app, claims)
        report["excluded_claim_attempts"].extend(bad_attempts)
        assert app["profiling_only"] and app["label_filter"] == label
        assert app["target_index"] == expected["target_index"]
        assert app["plan_sha256"] == expected["plan_sha256"]
        assert app["runner_sha256"] == queue["source_file_sha256"][str(OUT / "experiment_runner.py")]
        assert list(app["libraries"]) == [label]
        assert app["libraries"][label] == command["expected_library"]
        assert len(app["rows"]) == 1
        row = app["rows"][0]
        assert row["target_index"] == expected["target_index"]
        assert row["kid"] == expected["kid"] and row["shape"] == expected["shape"]
        assert row["split"] == 1 and row["profile_label"] == label and row["profile_iterations"] == 51
        assert app["gpu"]["pci_bdf"] == app["gpu"]["expected_pci_bdf"]
        source = next(folder.glob("pmc_1/*counter_collection.csv"))
        groups = parser.parse_file(source, re.compile("gemm_a8w8_mxfp8|opus_gemm_mxscale_bpreshuffle_reduce_kernel"), 51)
        assert len(groups) == 1
        parsed = groups[0]
        samples = parsed["samples"]
        assert len(samples) == 51
        assert len({(sample["identity"]["process"], sample["identity"]["agent"], sample["identity"]["queue"])
                    for sample in samples}) == 1
        for sample in samples:
            assert int(sample["metadata"]["Workgroup_Size"]) == expected["waves_per_workgroup"] * 64
            assert int(sample["metadata"]["Grid_Size"]) // int(sample["metadata"]["Workgroup_Size"]) == expected["workgroups"]
            if group == "sq_ea":
                counter = sample["counters"]
                assert counter["SQ_WAVES"] == expected["waves"]
                assert counter["SQ_INSTS_VALU_MFMA_MOPS_F8"] * 512 == expected["padded_f8_operations"]
                assert counter["SQ_VALU_MFMA_BUSY_CYCLES"] == expected["dynamic_scaled_mfma"] * 32
        identity = samples[0]["identity"]
        agent_id = int(identity["agent"].split()[-1])
        with next(folder.glob("pmc_1/*agent_info.csv")).open() as stream:
            agents = list(csv.DictReader(stream))
        agent = next(item for item in agents if int(item["Logical_Node_Id"]) == agent_id)
        location = int(agent["Location_Id"])
        pci = f"{int(agent['Domain']):04x}:{location >> 8:02x}:{(location >> 3) & 31:02x}.{location & 7}"
        assert pci == app["gpu"]["pci_bdf"]
        implied = [sample["counters"]["GRBM_COUNT"] / (sample["end_ns"] - sample["start_ns"]) for sample in samples]
        metrics = dict(parsed["median_metrics"])
        # Keep raw counters but exclude this invalid time-percentage alias.
        invalid_pct = metrics.pop("sq_vmem_addr_stall_pct_estimate", None)
        entry = {
            "name": name, "family": command["family"], "group": group, "label": label,
            "kid": expected["kid"], "shape": expected["shape"],
            "source_csv": str(source), "source_csv_sha256": sha(source),
            "application_sha256": sha(app_path), "plan_sha256": app["plan_sha256"],
            "library": app["libraries"][label], "runner_sha256": app["runner_sha256"],
            "claim_clean": True, "claim_window": {"start": claim["start"]["time"], "end": claim["end"]["time"]},
            "application_window": {"start": app["started"], "end": app["finished"]},
            "identity": identity, "kernel_name": parsed["kernel_name"],
            "agent": {"pci_bdf": pci, "CU": int(agent["Cu_Count"]), "SIMD": int(agent["Simd_Count"]),
                      "XCC": int(agent["Num_Xcc"]), "max_engine_fcompute_mhz": int(agent["Max_Engine_Clk_Fcompute"])},
            "dispatches_before_selection": parsed["dispatches_before_selection"], "selected_dispatches": 51,
            "profile_rotation": row["profile_rotation"], "expected_work": expected,
            "median_metrics": metrics, "median_counters": parsed["median_counters"],
            "GRBM_implied_GHz": {"min": min(implied), "median": statistics.median(implied), "max": max(implied)},
            "discarded_non_time_FIFO_percentage": invalid_pct,
        }
        report["collections"].append(entry)
    for family in {command["family"] for command in queue["commands"]}:
        for group in ("sq_ea", "ta_lds"):
            entries = {entry["label"]: entry for entry in report["collections"]
                       if entry["family"] == family and entry["group"] == group}
            if set(entries) != {"baseline", "candidate"}:
                continue
            baseline, candidate = entries["baseline"], entries["candidate"]
            assert baseline["agent"] == candidate["agent"], (family, group)
            assert baseline["shape"] == candidate["shape"] and baseline["profile_rotation"] == candidate["profile_rotation"]
            def delta_table(key):
                rows = {}
                for name in sorted(set(baseline[key]) & set(candidate[key])):
                    b, c = baseline[key][name], candidate[key][name]
                    rows[name] = {"baseline_median": b, "candidate_median": c,
                                  "candidate_minus_baseline": c - b,
                                  "candidate_minus_baseline_pct": 100 * (c / b - 1) if b else None}
                return rows
            report["comparisons"].append({
                "family": family, "group": group, "baseline_source": baseline["source_csv"],
                "candidate_source": candidate["source_csv"], "independent_passes_not_paired_dispatches": True,
                "median_metric_differences": delta_table("median_metrics"),
                "median_raw_counter_differences": delta_table("median_counters"),
                "diagnosis_status": "Directional clues only; no root-cause or adoption-speedup claim from this single independent pass pair.",
            })
    ROOT.mkdir(exist_ok=True)
    output = ROOT / "ab_mechanism_counter_summary.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "clean_collections": len(report["collections"]),
                      "completed_group_comparisons": len(report["comparisons"]), "pending": report["pending"],
                      "excluded": [entry["name"] for entry in report["excluded"]], "output": str(output)}))


if __name__ == "__main__":
    main()
