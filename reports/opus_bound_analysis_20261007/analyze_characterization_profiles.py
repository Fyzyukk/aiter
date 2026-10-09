#!/usr/bin/env python3
"""Review independent completed characterization PMC passes without GPU access."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import statistics

OUT = Path(__file__).resolve().parent
ROOT = OUT / "char_profiles"
GROUPS = ("sq_ea", "l2_tagmap", "utcl1_credits", "ta_lds")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    spec = importlib.util.spec_from_file_location("char_counter_parser", OUT / "parse_counters.py")
    parser = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parser)
    plan = json.loads((OUT / "plans/official_characterization.json").read_text())
    claims = [json.loads(line) for line in (OUT / "gpu_claim_log.jsonl").read_text().splitlines()]
    report = {
        "status": "cpu_review_no_gpu_access", "cross_pass_policy": "Each pass remains an independent source/dispatch set; only compare directions and separate medians.",
        "official_binary_sha256": plan["official_binary_sha256"], "collections": [],
        "excluded": [], "excluded_claim_attempts": [], "pending": [], "runtime_group_acceptance": {},
        "limits": [
            "Instrumented duration and physical DRAM throughput are observations, not unprofiled speedup or peak saturation proof.",
            "GRBM clock/window discrepancy remains for short wide9051. Do not apply a fixed multiplier or classify a bound from absolute utilization/occupancy.",
            "Scalar reductions discard channel/set/CU instance distribution; no hotspot can be established from these files.",
            "SQ VMEM address FIFO stall is unwindowed and does not share a proven scope with SQ_BUSY_CYCLES; ratios over100% are not time percentages.",
            "UTCL1 thrashing probes overlap; credit stalls may count without an effective request; zero measured aggregate does not establish zero cost.",
            "TA paper address ratio includes mixed buffer traffic; direct-LDS and width differ between the two kernels.",
        ],
    }
    expected = {
        30: {"kid": 9000, "shape": [8192, 8192, 8192], "workgroups": 1024,
             "waves": 4096, "MFMA": 16777216, "executed_ops": 1099511627776},
        31: {"kid": 9051, "shape": [1, 65536, 1536], "workgroups": 2048,
             "waves": 8192, "MFMA": 49152, "executed_ops": 3221225472},
    }
    for index, target in expected.items():
        for group in GROUPS:
            name = f"char_target{index}_kid{target['kid']}_{group}"
            folder = ROOT / name
            app_path = folder / "application.json"
            if not app_path.exists():
                report["pending"].append(name)
                continue
            app = json.loads(app_path.read_text())
            if app.get("status") != "passed" or app.get("contamination"):
                report["excluded"].append({"name": name, "status": app.get("status"),
                                          "contamination": app.get("contamination"), "reason": "No completed clean application evidence."})
                continue
            starts = [(i, row) for i, row in enumerate(claims)
                      if row.get("event") == "start" and row.get("command", {}).get("name") == name]
            clean_claim = None
            for i, start in reversed(starts):
                next_start = next((j for j in range(i + 1, len(claims))
                                   if claims[j].get("event") == "start"
                                   and claims[j].get("command", {}).get("name") == name), len(claims))
                end = next((row for row in claims[i + 1:next_start]
                            if row.get("event") == "end" and row.get("name") == name), None)
                if end and (end["returncode"] != 0 or end.get("contamination")):
                    report["excluded_claim_attempts"].append({
                        "name": name, "start": start["time"], "end": end["time"],
                        "returncode": end["returncode"], "contamination": end.get("contamination", False),
                        "reason": "Interrupted/contaminated attempt remains excluded even after a clean rerun overwrites current application files.",
                    })
                if (clean_claim is None and end and end["returncode"] == 0 and not end.get("contamination")
                        and start["time"] <= app["started"] <= app["finished"] <= end["time"]):
                    clean_claim = {"start": start, "end": end}
            assert clean_claim is not None, name
            assert not any(row.get("event") == "external_work_started"
                           and clean_claim["start"]["time"] <= row["time"] <= clean_claim["end"]["time"] for row in claims)
            app_row = app["rows"][0]
            assert app_row["kid"] == target["kid"] and app_row["shape"] == target["shape"]
            assert app_row["actual_official_module"]["sha256"] == plan["official_binary_sha256"]
            assert app["gpu"]["pci_bdf"] == app["gpu"]["expected_pci_bdf"]
            source = next(folder.glob("pmc_1/*counter_collection.csv"))
            groups = parser.parse_file(source, re.compile("gemm_a8w8_mxfp8|opus_gemm_mxscale_bpreshuffle_reduce_kernel"), 51)
            assert len(groups) == 1
            parsed_group = groups[0]
            samples = parsed_group["samples"]
            assert len(samples) == 51
            assert len({(s["identity"]["process"], s["identity"]["agent"], s["identity"]["queue"]) for s in samples}) == 1
            for sample in samples:
                metadata = sample["metadata"]
                assert int(metadata["Grid_Size"]) // int(metadata["Workgroup_Size"]) == target["workgroups"]
                if group == "sq_ea":
                    counts = sample["counters"]
                    assert counts["SQ_WAVES"] == target["waves"]
                    assert counts["SQ_INSTS_VALU_MFMA_MOPS_F8"] * 512 == target["executed_ops"]
                    assert counts["SQ_VALU_MFMA_BUSY_CYCLES"] == target["MFMA"] * 32
            identity = samples[0]["identity"]
            agent_id = int(identity["agent"].split()[-1])
            with next(folder.glob("pmc_1/*agent_info.csv")).open() as stream:
                agents = list(csv.DictReader(stream))
            agent = next(row for row in agents if int(row["Logical_Node_Id"]) == agent_id)
            location = int(agent["Location_Id"])
            pci = f"{int(agent['Domain']):04x}:{location >> 8:02x}:{(location >> 3) & 31:02x}.{location & 7}"
            assert pci == app["gpu"]["pci_bdf"]
            implied = [s["counters"]["GRBM_COUNT"] / (s["end_ns"] - s["start_ns"]) for s in samples]
            extra = {}
            counters = parsed_group["median_counters"]
            if group == "ta_lds":
                conflict = counters["SQ_LDS_BANK_CONFLICT"]
                extra["LDS_conflict_raw_median"] = conflict
                extra["SQ_address_stall_over_SQbusy_is_time_fraction"] = False
                if index == 30:
                    extra["LDS_conflict_per_dynamic_MFMA_count_units"] = conflict / target["MFMA"]
                extra["direct_LDS_buffer_reads"] = counters["TA_BUFFER_READ_LDS_WAVEFRONTS"]
            entry = {
                "name": name, "target_index": index, "group": group, "shape": target["shape"],
                "source_csv": str(source), "source_csv_sha256": sha(source),
                "application_sha256": sha(app_path), "claim_clean": True,
                "claim_window": {"start": clean_claim["start"]["time"], "end": clean_claim["end"]["time"]},
                "application_window": {"start": app["started"], "end": app["finished"]},
                "claim_fingerprint": clean_claim["start"].get("fingerprint"),
                "runtime_group_accepted": True, "selected_dispatches": 51,
                "dispatches_before_selection": parsed_group["dispatches_before_selection"],
                "identity": identity, "agent": {"pci_bdf": pci, "CU": int(agent["Cu_Count"]),
                    "SIMD": int(agent["Simd_Count"]), "XCC": int(agent["Num_Xcc"]),
                    "max_engine_fcompute_mhz": int(agent["Max_Engine_Clk_Fcompute"])},
                "median_metrics": parsed_group["median_metrics"],
                "median_counters": counters,
                "GRBM_implied_GHz": {"min": min(implied), "median": statistics.median(implied), "max": max(implied)},
                "extra_scope_notes": extra,
            }
            report["collections"].append(entry)
            report["runtime_group_acceptance"].setdefault(group, []).append(target["kid"])
    (ROOT / "characterization_counter_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "clean_collections": len(report["collections"]),
                      "excluded": [entry["name"] for entry in report["excluded"]], "pending": report["pending"]}))


if __name__ == "__main__":
    main()
