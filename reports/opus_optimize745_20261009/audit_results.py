#!/usr/bin/env python3
"""Audit persisted numerical, timing and selection evidence without using a GPU."""
import argparse
import collections
import csv
import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROTOCOLS = ("reuse1", "pool8")
NUMERICAL_PHASES = {"smoke", "fused_smoke"}
PERFORMANCE_PHASES = {"baseline", "tune", "tune_splits", "fused_tune", "replay"}
PHYSICAL_BDFS = {0: "0000:05:00.0", 1: "0000:15:00.0",
                 2: "0000:65:00.0", 3: "0000:75:00.0"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bdf(value):
    domain, bus, device = value.lower().split(":")
    slot, function = device.split(".")
    return f"{int(domain, 16):04x}:{int(bus, 16):02x}:{int(slot, 16):02x}.{int(function)}"


def errors_valid(checks):
    assert checks, "Missing numerical checks"
    assert all(math.isfinite(c["error_ratio"]) and 0 <= c["error_ratio"] <= .05 for c in checks)


def audit_row(row, phase, case):
    assert row["shape"] == [case[key] for key in ("M", "N", "K")]
    assert row["baseline"] == case["baseline"] and row["flydsl_name"] == case["flydsl_name"]
    if phase in NUMERICAL_PHASES or phase == "replay":
        assert not row["invalid"], "Numerical/replay worker contains rejected variants"
    pool_count = row["pool_count"]
    assert 2 <= pool_count <= 8
    tags = {task["config"] + ":" + str(task["split_k"]) for task in row["finalists"]}
    tags.add("flydsl_bmm")
    for tag in tags:
        checks = row["checks"][tag]
        errors_valid(checks)
        assert {c["repeat"] for c in checks if c["seed"] == 29} == {0, 1, 2}
        assert {c["pool_index"] for c in checks if c["seed"] == 101} == set(range(pool_count))
        if tag != "flydsl_bmm" and not tag.startswith("fused_"):
            public = row["public_api_checks"][tag]
            errors_valid(public)
            assert {c["seed"] for c in public} == {29, 101}, "Missing finalist public API seed"
    if phase in NUMERICAL_PHASES:
        assert not row["records"] and "medians" not in row
        return None
    assert phase in PERFORMANCE_PHASES
    records = {(r["variant"], r["protocol"], r["round"]): r for r in row["records"]}
    expected = {(tag, protocol, i) for tag in tags for protocol in PROTOCOLS for i in range(6)}
    assert len(records) == len(row["records"]) and set(records) == expected, "Timing round coverage"
    for (tag, protocol, _), record in records.items():
        assert math.isfinite(record["us"]) and record["us"] > 0
        ratios = record["post_timing_error_ratios"]
        assert len(ratios) == (1 if protocol == "reuse1" else pool_count)
        assert all(math.isfinite(r) and 0 <= r <= .05 for r in ratios)
    for protocol in PROTOCOLS:
        assert set(row["medians"][protocol]) == tags
        for tag in tags:
            recomputed = statistics.median(records[tag, protocol, i]["us"] for i in range(6))
            assert row["medians"][protocol][tag] == recomputed, "Stored median differs from raw rounds"
    choices = []
    # Preserve the watcher's median insertion order when exact timing ties occur.
    for tag in row["medians"]["pool8"]:
        if tag == "flydsl_bmm":
            continue
        rounds = {p: sum(records[tag, p, i]["us"] < records["flydsl_bmm", p, i]["us"]
                         for i in range(6)) for p in PROTOCOLS}
        speedup = {p: row["medians"][p]["flydsl_bmm"] / row["medians"][p][tag] for p in PROTOCOLS}
        both = all(s > 1 for s in speedup.values())
        choices.append({"tag": tag, "stable_win": both and all(r >= 5 for r in rounds.values()),
                        "both_protocols_faster": both, "winning_rounds": rounds, "speedup": speedup})
    return min(choices, key=lambda c: (not c["stable_win"], not c["both_protocols_faster"],
                                       row["medians"]["pool8"][c["tag"]]))


def audit_receipt(path, receipt, plan, plan_sha, benchmark_sha, helper_sha):
    output = Path(receipt["output"])
    cases_path = output.with_suffix(".cases.json")
    assert receipt["returncode"] == 0 and not receipt["contaminated"] and not receipt["external"]
    assert receipt["plan_sha256"] == plan_sha and receipt["script_sha256"] == benchmark_sha
    assert sha(output) == receipt["output_sha256"] and sha(cases_path) == receipt["cases_sha256"]
    result = json.loads(output.read_text())
    phase = result["phase"]
    assert result["status"] == "completed"
    for key, expected in (("plan_sha256", plan_sha), ("script_sha256", benchmark_sha),
                          ("cases_sha256", receipt["cases_sha256"])):
        assert result[key] == expected
    gpu = receipt["gpu"]
    assert gpu["smi_index"] in plan["allowed_smi_indices"]
    assert bdf(gpu["bdf"]) == PHYSICAL_BDFS[gpu["smi_index"]] == bdf(result["gpu_bdf"])
    assert len(gpu["gfx"]) == 3 and max(gpu["gfx"]) <= 2 and gpu["used_gib"] <= 8
    assert not gpu["processes"] and receipt["owner"]["inner_pid"] == result["pid"]
    assert receipt["owner"]["new_host_pids"] == [receipt["owner"]["host_pid"]]
    if phase in PERFORMANCE_PHASES:
        assert receipt.get("kfd_helper_sha256") == helper_sha and gpu.get("kfd_owners") == []
        known_watchers = {sha(HERE / "watch.py")}
        known_watchers.update(sha(p) for p in (HERE / "harness_revisions").glob("watch_v*.py"))
        assert receipt.get("watcher_sha256") in known_watchers
    for name, digest in receipt.get("dependency_sha256", {}).items():
        assert sha(name) == digest
    for module in result["modules"].values():
        assert sha(module["path"]) == module["sha256"]
    cases = json.loads(cases_path.read_text())
    assert len(cases) == len(result["rows"])
    plan_cases = {(c["M"], c["N"], c["K"]): c for c in plan["cases"]}
    rows = []
    for row, case in zip(result["rows"], cases):
        original = plan_cases[tuple(row["shape"])]
        assert case["baseline"] == original["baseline"] and case["flydsl_name"] == original["flydsl_name"]
        picked = audit_row(row, phase, case)
        if phase == "replay":
            assert len(case["replay"]) == 1
            task = case["replay"][0]
            assert task["config"] + ":" + str(task["split_k"]) in row["checks"]
        rows.append({"shape": row["shape"], "picked": picked, "receipt": str(path), "row": row})
    return phase, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    from csrc.opus_gemm.opus_gemm_bpreshuffle_config import config_identity
    from csrc.opus_gemm.opus_gemm_bpreshuffle_policy import select_config
    plan = json.loads((HERE / "plan.json").read_text())
    for name, digest in {**plan["source_sha256"], **plan["input_sha256"]}.items():
        assert sha(ROOT / name) == digest, f"Frozen source/input mismatch: {name}"
    plan_sha, benchmark_sha, helper_sha = (sha(HERE / name) for name in ("plan.json", "benchmark.py", "kfd_owners.py"))
    evidence = collections.defaultdict(dict)
    excluded, stale, failures = [], [], []
    for path in sorted(HERE.glob("*.receipt.json")):
        receipt = json.loads(path.read_text())
        if not receipt["valid_for_analysis"]:
            excluded.append(str(path))
            continue
        if receipt["plan_sha256"] != plan_sha or receipt["script_sha256"] != benchmark_sha:
            stale.append(str(path))
            continue
        try:
            phase, rows = audit_receipt(path, receipt, plan, plan_sha, benchmark_sha, helper_sha)
            for row in rows:
                evidence[phase][tuple(row["shape"])] = row
        except (AssertionError, KeyError, OSError, ValueError) as error:
            failures.append({"receipt": str(path), "error": type(error).__name__ + ": " + str(error)})
    expected_shapes = {(c["M"], c["N"], c["K"]) for c in plan["cases"]}
    assert len(expected_shapes) == 745
    final = evidence["replay"]
    stable_wins = sum(v["picked"]["stable_win"] for v in final.values())
    private_selections = sum(v["picked"]["tag"].startswith("fused_") for v in final.values())
    selected_path = HERE / "selected745.csv"
    selection_audited = False
    if selected_path.exists():
        try:
            config_lookup = dict(plan["configs"])
            private_lookup = {}
            for directory, prefix in (("fused_register", "fused_"), ("fused_lds", "fused_lds_"),
                                      ("fused_wavek", "fused_wavek_")):
                for config in json.loads((HERE / directory / "build.json").read_text())["configurations"]:
                    config_lookup.setdefault(config["identity"], config)
                    private_lookup[prefix + str(config["id"])] = config
            selected = list(csv.DictReader(selected_path.open()))
            assert len(selected) == len(final) == 745
            selected_shapes = {(int(r["M"]), int(r["N"]), int(r["K"])) for r in selected}
            assert selected_shapes == set(final) == expected_shapes
            for selected_row in selected:
                s = tuple(int(selected_row[k]) for k in ("M", "N", "K"))
                item = final[s]; picked = item["picked"]
                identity, split = picked["tag"].split(":")
                assert int(selected_row["splitK"]) == int(split)
                assert selected_row["stable_win"] == str(picked["stable_win"])
                assert float(selected_row["us"]) == item["row"]["medians"]["pool8"][picked["tag"]]
                assert float(selected_row["flydsl_bmm_us"]) == item["row"]["medians"]["pool8"]["flydsl_bmm"]
                assert float(selected_row["speedup_pool8"]) == picked["speedup"]["pool8"]
                assert selected_row["gfx"] == "gfx950" and int(selected_row["cu_num"]) == 256
                if identity.startswith("fused_"):
                    assert selected_row["libtype"] == "opus_private" and int(selected_row["kernelId"]) == -1
                    config = private_lookup[identity]
                    assert selected_row["pipeline"] == "private_" + config["pipeline"] + "_fused"
                    assert json.loads(selected_row["config"]) == config["compile_params"]
                else:
                    config = config_lookup[identity]
                    assert selected_row["libtype"] == "opus" and selected_row["pipeline"] == config["pipeline"]
                    assert json.loads(selected_row["config"]) == config["compile_params"]
                    selected_config = select_config(s, gfx="gfx950", cu_num=256, tuned_file=str(selected_path))
                    assert selected_config.source == "tuned" and config_identity(selected_config.config) == identity
                    assert selected_config.split_k == int(split), "CSV split changed during real policy lookup"
            summary = json.loads((HERE / "summary.json").read_text())
            assert summary["shapes"] == 745 and summary["stable_wins"] == stable_wins
            assert summary["private_selections_requiring_integration"] == private_selections
            selection_audited = True
        except (AssertionError, KeyError, OSError, ValueError, StopIteration) as error:
            failures.append({"selection": str(selected_path), "error": type(error).__name__ + ": " + str(error)})
    candidate_evidence_complete = (not failures and selection_audited and set(final) == expected_shapes
                                   and stable_wins == 745 and private_selections == 0)
    numerical = [v for phase in NUMERICAL_PHASES for v in evidence[phase].values()]
    report = {"status": "failed" if failures else "complete_candidate_evidence" if candidate_evidence_complete else "incomplete_evidence",
              "recorded_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "target_completed": False, "candidate_evidence_complete": candidate_evidence_complete,
              "production_integration_and_tuned_dispatch_replay": "requires separate final validation",
              "gpu_queries": 0, "gpu_contexts": 0, "kernel_launches": 0,
              "plan_sha256": plan_sha, "benchmark_sha256": benchmark_sha,
              "kfd_helper_sha256": helper_sha, "phases": {p: len(v) for p, v in sorted(evidence.items())},
              "numerical_variants": sum(len(v["row"]["checks"]) for v in numerical),
              "public_api_variants": sum(len(v["row"]["public_api_checks"]) for v in numerical),
              "signed_checks": sum(len(checks) for v in numerical for checks in v["row"]["checks"].values()),
              "replay_shapes": len(final), "stable_wins": stable_wins,
              "private_selections_requiring_integration": private_selections,
              "selected_csv_audited": selection_audited,
              "missing_replay_shapes": sorted(expected_shapes - set(final)),
              "excluded_receipts": excluded, "stale_receipts": stale, "failures": failures}
    if args.output:
        with args.output.open("x") as stream:
            stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "missing_replay_shapes"}))
    return 1 if failures else 0 if candidate_evidence_complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
