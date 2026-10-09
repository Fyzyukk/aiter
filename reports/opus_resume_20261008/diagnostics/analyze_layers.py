#!/usr/bin/env python3
"""CPU-only identity and layer analysis of four independent Oct 8 PMC passes.

Read all raw CSV/JSON records, select the last 51 matching dispatches, validate
their provenance and compute only explicitly scoped same-pass ratios. Never
import HIP, torch, amdsmi or a GPU profiler. Never normalize GRBM values.
"""
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
from urllib.parse import unquote

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OLD = ROOT / "reports/opus_bound_analysis_20261007"
GROUPS = ["l2_tagmap", "utcl1_credits", "latency", "lds_issue"]
LAST = 51


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def canonical(name):
    for prefix in ["OPUS_GFX950_", "OCT8_"]:
        if name.startswith(prefix):
            return name[len(prefix):].rsplit("_", 1)[0]
    return name


def summary(values):
    assert values and all(math.isfinite(v) for v in values)
    return {"min": min(values), "median": statistics.median(values), "max": max(values)}


def pci(domain, location):
    location, domain = int(location), int(domain)
    return f"{domain:04x}:{location >> 8:02x}:{(location >> 3) & 31:02x}.{location & 7}"


def fingerprint(command, environment):
    argv = command["argv"]
    output_index = argv.index("--output") + 1
    paths = [Path(v) for i, v in enumerate(argv) if i != output_index and Path(v).is_file()]
    paths.append(OLD / "experiment_runner.py")
    plan = json.loads(Path(argv[argv.index("--plan") + 1]).read_text())
    paths.extend(Path(v) for v in plan["libraries"].values())
    files = {str(p.resolve()): sha(p) for p in paths if p.is_file()}
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    diff = subprocess.check_output(["git", "diff", "HEAD", "--", "csrc/opus_gemm"], cwd=ROOT)
    record = {"argv": argv, "env": environment, "files": files,
              "source_head": head, "source_diff_sha256": digest(diff)}
    return digest(json.dumps(record, sort_keys=True).encode())


def clean_claim(command, app, claims, environment):
    matches = []
    for i, start in enumerate(claims):
        if start.get("event") != "start" or start.get("command", {}).get("name") != command["name"]:
            continue
        next_start = next((j for j in range(i + 1, len(claims))
                           if claims[j].get("event") == "start"), len(claims))
        window = claims[i + 1:next_start]
        ends = [r for r in window if r.get("event") == "end" and r.get("name") == command["name"]]
        if len(ends) != 1:
            continue
        end = ends[0]
        if end["returncode"] == 0 and not end.get("contamination") and start["time"] <= app["started"] <= app["finished"] <= end["time"]:
            matches.append((start, end, [r for r in window if start["time"] <= r["time"] <= end["time"]]))
    assert len(matches) == 1, (command["name"], "Expected one clean enclosing claim")
    start, end, window = matches[0]
    assert start["command"] == command
    assert start["fingerprint"] == fingerprint(command, environment)
    assert not any(r["event"] in ["external_work_started", "interrupted", "contamination"] for r in window)
    owners = [r for r in window if r["event"] == "owner_identity" and r.get("name") == command["name"]]
    assert len(owners) == 1
    owner = owners[0]
    assert owner["new_host_pids"] == [owner["host_pid"]]
    assert owner["launcher_sha256"] == sha(OLD / "owned_python_launch.py")
    monitors = [r for r in window if r["event"] == "monitor"]
    assert monitors
    for monitor in monitors:
        assert monitor["child_pid"] == owner["inner_pid"]
        assert all(p["pid"] == owner["host_pid"] for p in monitor["processes"])
    claimed = next(r for r in reversed(claims[:claims.index(start)]) if r["event"] == "claimed")
    assert claimed["gpu"]["bdf"] == app["gpu"]["pci_bdf"]
    return {"clean": True, "start": start, "end": end, "owner": owner,
            "monitor_records": monitors, "claimed_device": claimed["gpu"]}


def append_ratio(metrics, samples, name, numerator, denominator, unit, meaning, limitations, factor=1):
    values = [factor * s["canonical_counters"][numerator] / s["canonical_counters"][denominator]
              for s in samples]
    metrics[name] = {"formula": f"{factor} * {numerator} / {denominator}", "unit": unit,
                     "meaning": meaning, "limitations": limitations, "statistics": summary(values)}
    for sample, value in zip(samples, values):
        sample["matched_ratios"][name] = value


def ratios(group, samples):
    metrics = {}
    if group == "l2_tagmap":
        for sample in samples:
            c = sample["canonical_counters"]
            c["TCC_LOOKUPS"] = c["TCC_HIT"] + c["TCC_MISS"]
            c["TCP_TAGRAM_REQUESTS"] = sum(c[f"TCP_TAGRAM{i}_REQ"] for i in range(4))
        append_ratio(metrics, samples, "tcc_hit_request_fraction", "TCC_HIT", "TCC_LOOKUPS", "fraction of counted lookup events",
            "Same-pass cache hit/(hit+miss) request fraction.", ["UC reads count as misses; not a HBM byte fraction."])
        for i in range(4):
            append_ratio(metrics, samples, f"tcp_tagram{i}_request_fraction", f"TCP_TAGRAM{i}_REQ", "TCP_TAGRAM_REQUESTS",
                "fraction of counted TCP-to-TCC tagram requests", "Share in this aggregated tagram index.",
                ["Scalar sum discards per-TCP/TCC/channel/set distribution; unequal shares do not establish a hotspot."])
        append_ratio(metrics, samples, "ta_buffer_cycles_per_buffer_wavefront", "TA_BUFFER_TOTAL_CYCLES", "TA_BUFFER_WAVEFRONTS",
            "TA buffer-cycle events / buffer-wavefront event", "Count ratio for TA processing of buffer requests.",
            ["Requests include matrix and scale traffic; not kernel wall time, service latency or pipeline utilization."])
        append_ratio(metrics, samples, "tcc_tag_stall_per_busy_count", "TCC_TAG_STALL", "TCC_BUSY", "raw event ratio",
            "Contextual tag-pipeline stall/busy ratio within the same TCC pass.",
            ["Busy is not windowable and probes stall at multiple pipeline points; not a strict time fraction."])
    elif group == "utcl1_credits":
        for counter, name in [("TCP_UTCL1_SERIALIZATION_STALL", "utcl1_serialization_per_gate_count"),
                              ("TCP_UTCL1_THRASHING_STALL", "utcl1_thrashing_per_gate_count"),
                              ("TCP_UTCL1_STALL_INFLIGHT_MAX", "utcl1_inflight_max_per_gate_count")]:
            append_ratio(metrics, samples, name, counter, "TCP_GATE_EN1", "raw TCP counter ratio",
                "Contextual same-TCP counter ratio to interface-on count.",
                ["TCP_GATE_EN1 is not windowed; not a GEMM wall-time fraction.",
                 "Thrashing overlap/probe limitations apply to the thrashing counter."])
        append_ratio(metrics, samples, "ta_downstream_stall_per_busy_count", "TA_ADDR_STALLED_BY_TC_CYCLES", "TA_TA_BUSY",
            "raw TA counter ratio", "Contextual TA address-path downstream stall/busy ratio.",
            ["Both counters lack perf windowing; cannot assign the ratio as elapsed-time loss or uniquely to a downstream block."])
    elif group == "latency":
        append_ratio(metrics, samples, "ea_read_level_per_completed_request", "TCC_EA0_RDREQ_LEVEL", "TCC_EA0_RDREQ",
            "nominal TCC counter clock-cycles / read request", "SDK-defined EA inflight request-cycle integral divided by request count.",
            ["Endpoint/window and aggregation assumptions remain; not independently calibrated nanoseconds.",
             "EA read path includes external/cache service and is not a direct HBM-only latency measurement."])
        append_ratio(metrics, samples, "tcp_tcc_latency_per_read_request", "TCP_TCC_READ_REQ_LATENCY", "TCP_TCC_READ_REQ",
            "nominal TCP counter clock-cycles / read request", "Same-pass accumulated TCP-to-TCC latency divided by read requests.",
            ["Latency counter is not windowed and includes atomics with return; audited target ISA contains no atomic instruction.",
             "Boundary and clock assumptions prevent precise wall-time attribution; do not compare cycles across different clock domains."])
    elif group == "lds_issue":
        append_ratio(metrics, samples, "mean_wave_life_cycles", "SQ_WAVE_CYCLES", "SQ_WAVES", "wave clock-cycles / wave",
            "Mean cumulative wave life; SQ_WAVE_CYCLES has quad-cycle units.",
            ["Wave overlap prevents interpreting cumulative life as kernel wall time."], factor=4)
        for numerator, name, meaning in [("SQ_WAIT_ANY", "wave_any_wait_fraction", "Wave quad-cycles waiting for dependencies of any category."),
            ("SQ_WAIT_INST_ANY", "wave_issue_wait_fraction", "Wave quad-cycles waiting for instruction issue of any category."),
            ("SQ_WAIT_INST_LDS", "wave_lds_issue_wait_fraction", "Wave quad-cycles waiting specifically for LDS instruction issue."),
            ("SQ_ACTIVE_INST_LDS", "wave_lds_active_fraction", "Wave quad-cycles working on LDS instructions.")]:
            append_ratio(metrics, samples, name, numerator, "SQ_WAVE_CYCLES", "fraction of cumulative wave quad-cycles", meaning,
                ["Same-scope wave cumulative counters; not a wall-time fraction.",
                 "LDS issue waiting does not measure LDS dependency/return waits; ANY does not split VMEM/VALU/LDS/barrier dependencies."])
        append_ratio(metrics, samples, "lds_share_of_issue_wait", "SQ_WAIT_INST_LDS", "SQ_WAIT_INST_ANY",
            "fraction of cumulative issue-wait quad-cycles", "LDS-specific portion of counted instruction-issue waiting.",
            ["Does not isolate LDS bank-conflict delay or all LDS-related dependency waiting."])
        append_ratio(metrics, samples, "lds_instructions_per_wave", "SQ_INSTS_LDS", "SQ_WAVES",
            "issued LDS/FLAT instructions / wave", "SDK-issued LDS instruction event count divided by waves.",
            ["Includes FLAT by definition; instruction events are not bytes or elapsed cycles."])
    return metrics


def analyze(command, queue, plan, claims, binary_audit, definitions):
    name, group = command["name"], command["group"]
    folder = HERE / name
    paths = {"application": folder / "application.json", "results": folder / f"{name}_results.json",
             "counter_csv": folder / f"{name}_counter_collection.csv", "trace_csv": folder / f"{name}_kernel_trace.csv",
             "agent_csv": folder / f"{name}_agent_info.csv"}
    hashes = {key: sha(path) for key, path in paths.items()}
    app = json.loads(paths["application"].read_text())
    assert app["status"] == "passed" and not app.get("contamination")
    assert app["plan"] == plan and app["plan_sha256"] == sha(HERE / "plan.json")
    assert app["runner_sha256"] == sha(OLD / "experiment_runner.py")
    assert app["source_head"] == subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    assert app["profiling_only"] and app["label_filter"] == "selected" and app["target_index"] == command["target_index"] == 0
    assert not app["event_confirmation_requested"] and app["gpu"]["pci_bdf"] == app["gpu"]["expected_pci_bdf"]
    assert app["gpu"]["pci_bdf"] == "0000:65:00.0"
    assert len(app["rows"]) == 1 and list(app["libraries"]) == ["selected"]
    lib = Path(plan["libraries"]["selected"])
    assert app["libraries"]["selected"] == {"path": str(lib), "sha256": sha(lib)}
    row = app["rows"][0]
    assert row["shape"] == plan["targets"][0]["shape"] == [480, 7168, 384] and row["kid"] == 9021
    assert row["split"] == 1 and row["target_index"] == 0 and row["profile_label"] == "selected"
    assert row["profile_iterations"] == row["profile_rotation"] == LAST and row["profile_warmup"] == 5
    argv = command["argv"]
    assert Path(argv[argv.index("--plan") + 1]) == HERE / "plan.json"
    assert Path(argv[argv.index("--output") + 1]) == paths["application"]
    assert argv[argv.index("--label") + 1] == "selected" and argv[argv.index("--target-index") + 1] == "0"
    assert argv[argv.index("--iters") + 1] == "51" and "--profile" in argv
    extra = command["env"]["ROCPROF_EXTRA_COUNTERS_CONTENTS"]
    assert extra == (HERE / "extra_counters.yaml").read_text()
    aliases = {c["name"]: c for c in yaml.safe_load(extra)["rocprofiler-sdk"]["counters"]}
    requested = set(command["env"]["ROCPROF_COUNTERS"].removeprefix("pmc: ").split())
    claim = clean_claim(command, app, claims, {**queue["env"], **command["env"]})
    data = json.loads(paths["results"].read_text())["rocprofiler-sdk-tool"]
    assert len(data) == 1
    data = data[0]
    assert data["metadata"]["pid"] == claim["owner"]["inner_pid"]
    assert data["metadata"]["command"] == [argv[0], str(OLD / "owned_python_launch.py"), *argv[1:]]
    target = [k for k in data["kernel_symbols"] if "gemm_a8w8_mxfp8_scale_4wave_128x128_kernel" in k.get("kernel_name", "")]
    assert len(target) == 1
    target = target[0]
    current = next(x for x in binary_audit["libraries"] if x["label"] == "baseline")
    assert current["input_sha256"] == sha(lib)
    kernel = next(k for k in current["kernels"] if k["kid"] == 9021)
    assert target["kernel_name"] == kernel["name"] + ".kd" and target["formatted_kernel_name"] == kernel["demangled"]
    assert target["kernarg_segment_size"] == kernel["metadata"][".kernarg_segment_size"] == 96
    assert target["group_segment_size"] == kernel["metadata"][".group_segment_fixed_size"] == 105504
    assert target["private_segment_size"] == 0
    assert not any("atomic" in k for k in kernel["instruction_counts"])
    objects = [o for o in data["code_objects"] if o["code_object_id"] == target["code_object_id"]]
    assert len(objects) == 1
    obj = objects[0]
    match = re.fullmatch(r"file://(.+)#offset=(\d+)&size=(\d+)", obj["uri"])
    assert match and Path(unquote(match[1])) == lib
    offset, size = int(match[2]), int(match[3])
    image = lib.read_bytes()[offset:offset + size]
    assert len(image) == size and digest(image) == current["device_sha256"]
    agents = read_csv(paths["agent_csv"])
    sdk_agent = next(a for a in data["agents"] if a["id"]["handle"] == obj["agent_id"]["handle"])
    agent = next(a for a in agents if int(a["Logical_Node_Id"]) == sdk_agent["logical_node_id"])
    assert pci(agent["Domain"], agent["Location_Id"]) == pci(sdk_agent["domain"], sdk_agent["location_id"]) == app["gpu"]["pci_bdf"]
    assert int(agent["Cu_Count"]) == sdk_agent["cu_count"] == 256 and int(agent["Simd_Count"]) == sdk_agent["simd_count"] == 1024
    definitions_by_id = {}
    for c in data["counters"]:
        cid = c["id"]["handle"]
        if cid in definitions_by_id:
            assert definitions_by_id[cid]["name"] == c["name"]
        definitions_by_id[cid] = c
        if c["name"] in requested and c["name"] in aliases:
            expected_alias = aliases[c["name"]]
            definition = next(d for d in expected_alias["definitions"] if "gfx950" in d["architectures"])
            assert c["expression"] == definition["expression"] and c["is_derived"] == 1
    assert all(canonical(n) in definitions or n.startswith("OPUS_GFX950_") or n in ["GRBM_COUNT", "GRBM_GUI_ACTIVE"] for n in requested)
    raw_rows = read_csv(paths["counter_csv"])
    assert {r["Counter_Name"] for r in raw_rows} == requested
    callbacks = data["callback_records"]["counter_collection"]
    assert len(callbacks) == 57
    by_dispatch = {str(c["dispatch_data"]["dispatch_info"]["dispatch_id"]): c for c in callbacks}
    assert len(by_dispatch) == 57
    grouped = collections.defaultdict(list)
    for line, raw in enumerate(raw_rows, 2):
        assert raw["Kernel_Name"] == target["formatted_kernel_name"]
        grouped[raw["Dispatch_Id"]].append({"csv_line": line, **raw})
    assert set(grouped) == set(by_dispatch)
    trace_rows = read_csv(paths["trace_csv"])
    matching_trace = [r for r in trace_rows if r["Kernel_Name"] == target["formatted_kernel_name"]]
    assert len(matching_trace) == 57 and {r["Dispatch_Id"] for r in matching_trace} == set(by_dispatch)
    trace_by_dispatch = {r["Dispatch_Id"]: r for r in matching_trace}
    full_samples = []
    for dispatch, csv_rows in grouped.items():
        assert len(csv_rows) == len(requested) and {r["Counter_Name"] for r in csv_rows} == requested
        first, callback, trace = csv_rows[0], by_dispatch[dispatch], trace_by_dispatch[dispatch]
        d, info = callback["dispatch_data"], callback["dispatch_data"]["dispatch_info"]
        identity_keys = ["Process_Id", "Agent_Id", "Queue_Id", "Dispatch_Id", "Kernel_Id", "Correlation_Id",
                         "Start_Timestamp", "End_Timestamp", "Kernel_Name", "Grid_Size", "Workgroup_Size",
                         "LDS_Block_Size", "Scratch_Size", "VGPR_Count", "SGPR_Count", "Accum_VGPR_Count"]
        assert all(all(r[k] == first[k] for k in identity_keys) for r in csv_rows)
        assert int(first["Process_Id"]) == data["metadata"]["pid"]
        assert first["Agent_Id"] == f"Agent {sdk_agent['logical_node_id']}" and int(first["Queue_Id"]) == info["queue_id"]["handle"]
        assert int(first["Kernel_Id"]) == info["kernel_id"] == target["kernel_id"]
        assert int(first["Correlation_Id"]) == d["correlation_id"]["internal"]
        start, end = int(first["Start_Timestamp"]), int(first["End_Timestamp"])
        assert start == d["start_timestamp"] == int(trace["Start_Timestamp"])
        assert end == d["end_timestamp"] == int(trace["End_Timestamp"]) and end > start
        assert math.prod(info["workgroup_size"].values()) == int(first["Workgroup_Size"]) == 256
        assert math.prod(info["grid_size"].values()) == int(first["Grid_Size"]) == 224 * 256
        assert info["private_segment_size"] == int(first["Scratch_Size"]) == 0 and info["group_segment_size"] == 105504
        assert int(trace["Workgroup_Size_X"]) * int(trace["Workgroup_Size_Y"]) * int(trace["Workgroup_Size_Z"]) == 256
        assert int(trace["Grid_Size_X"]) * int(trace["Grid_Size_Y"]) * int(trace["Grid_Size_Z"]) == 224 * 256
        assert all(first[k] == trace[k] for k in ["Agent_Id", "Queue_Id", "Kernel_Id", "Correlation_Id",
                                                 "LDS_Block_Size", "Scratch_Size", "VGPR_Count", "SGPR_Count", "Accum_VGPR_Count"])
        emitted = collections.defaultdict(list)
        for record in callback["records"]:
            emitted[definitions_by_id[record["counter_id"]["handle"]]["name"]].append(record["value"])
        assert set(emitted) == requested
        scalar = {r["Counter_Name"]: float(r["Counter_Value"]) for r in csv_rows}
        for n, values in emitted.items():
            assert all(math.isfinite(v) for v in values)
            assert len(values) == (8 if n in ["GRBM_COUNT", "GRBM_GUI_ACTIVE"] else 1)
            assert math.isclose(scalar[n], sum(values), rel_tol=0, abs_tol=1e-5)
        full_samples.append({"identity": {k: first[k] for k in identity_keys}, "start_ns": start, "end_ns": end,
            "profiled_duration_ns": end - start, "csv_rows": csv_rows, "json_callback_record": callback,
            "trace_record": trace, "raw_csv_counter_values": scalar,
            "raw_callback_values_in_emitted_order": dict(emitted),
            "canonical_counters": {canonical(k): v for k, v in scalar.items()}, "matched_ratios": {}})
    full_samples.sort(key=lambda s: (s["start_ns"], int(s["identity"]["Dispatch_Id"])))
    samples = full_samples[-LAST:]
    assert len(samples) == LAST
    assert len({tuple(s["identity"][k] for k in ["Process_Id", "Agent_Id", "Queue_Id", "Kernel_Id"]) for s in full_samples}) == 1
    if group == "lds_issue":
        assert all(s["canonical_counters"]["SQ_WAVES"] == 896 for s in full_samples)
    metrics = ratios(group, samples)
    medians = {name: statistics.median(s["raw_csv_counter_values"][name] for s in samples) for name in requested}
    consumed = {"application": app, "target_runtime_symbol": target, "loaded_code_object": obj,
        "sdk_agent": sdk_agent, "csv_agent": agent, "requested_runtime_counter_definitions":
        {n: next(c for c in data["counters"] if c["name"] == n) for n in requested}}
    assert hashes == {key: sha(path) for key, path in paths.items()}, "Raw input changed during analysis"
    return {"name": name, "group": group, "kid": 9021, "shape": [480, 7168, 384],
        "input_paths": {k: str(v) for k, v in paths.items()}, "input_sha256": hashes, "inputs_unchanged": True,
        "strict_identity_validation_passed": True, "claim": claim, "provenance": consumed,
        "loaded_device_sha256": digest(image), "audited_instruction_sha256": kernel["instruction_sha256"],
        "independent_pass_not_a_cross_group_dispatch_pair": True,
        "dispatches_before_selection": len(full_samples), "selected_last_dispatches": LAST,
        "excluded_initial_dispatch_ids": [s["identity"]["Dispatch_Id"] for s in full_samples[:-LAST]],
        "selected_dispatch_ids": [s["identity"]["Dispatch_Id"] for s in samples],
        "profile_rotation": row["profile_rotation"], "profiled_duration_ns": summary([s["profiled_duration_ns"] for s in samples]),
        "matched_ratio_metrics": metrics, "median_raw_csv_counter_values": medians,
        "all_raw_dispatch_samples": full_samples, "raw_counter_csv_rows": raw_rows,
        "raw_grbm_policy": "CSV raw GRBM sums and JSON eight-value emitted sequences are preserved. No max/sum correction, clock normalization, occupancy, MFMA utilization or cycle-to-nanosecond conversion is applied. Callback raw records do not identify each XCC index explicitly."}


def main():
    queue = json.loads((HERE / "queue.json").read_text())
    plan = json.loads((HERE / "plan.json").read_text())
    claims = [json.loads(line) for line in (HERE / "gpu_claim_log.jsonl").read_text().splitlines()]
    binary_path = HERE.parent / "sfa_packed/device_audit.json"
    binary = json.loads(binary_path.read_text())
    assert binary["status"] == "passed" and binary["summary"]["baseline_matches_oct7_selected"]
    definitions = json.loads((HERE / "counter_definitions.json").read_text())
    extra = yaml.safe_load((HERE / "extra_counters.yaml").read_text())
    commands = [c for c in queue["commands"] if c["group"] in GROUPS]
    assert len(commands) == 4 and {c["group"] for c in commands} == set(GROUPS)
    collections_ = [analyze(c, queue, plan, claims, binary, definitions) for c in commands]
    report = {"status": "passed_cpu_identity_and_matched_layer_review", "cpu_only": True,
        "gpu_imported_or_initialized": False, "gpu_executed_by_analyzer": False, "analyzer_sha256": sha(Path(__file__)),
        "target": {"kid": 9021, "shape": [480, 7168, 384], "workgroups": 224, "waves": 896,
                   "pci_bdf": "0000:65:00.0", "implementation": "Oct 7 retained prologue; original raw-byte SFA LDS layout"},
        "source_sha256": {str(p): sha(p) for p in [HERE / "queue.json", HERE / "plan.json", HERE / "gpu_claim_log.jsonl",
            HERE / "counter_definitions.json", HERE / "extra_counters.yaml", binary_path, OLD / "experiment_runner.py", OLD / "owned_python_launch.py"]},
        "counter_definitions": definitions, "extra_counter_aliases": extra, "collections": collections_,
        "summary": {"clean_independent_passes": 4, "selected_dispatches_per_pass": LAST,
                    "total_selected_dispatches": 4 * LAST, "raw_dispatches_preserved": sum(c["dispatches_before_selection"] for c in collections_),
                    "all_identity_checks_passed": True, "all_raw_inputs_unchanged": True},
        "limits": ["One independent pass per counter group; 51 dispatches describe one process/address-rotation session, not 51 independent experiments.",
            "Cross-group scalar medians are separate observations and must not be joined into a synthetic dispatch or cumulative bottleneck budget.",
            "Profiler duration is instrumented and does not establish unprofiled Event performance or a speedup.",
            "Raw GRBM values are preserved without normalization. No absolute pipeline utilization, occupancy or cycle-to-time conversion.",
            "LDS issue wait measures instruction-issue waiting, not LDS data dependency/return waiting. SQ_WAIT_ANY cannot identify VMEM, LDS, VALU or barriers.",
            "Tagram shares discard channel/set/TCP instance distributions; no hotspot is established.",
            "Credit and LFIFO raw counters have no safe matched wall-time denominator here. Non-windowed gate/busy/latency ratios are contextual count ratios only.",
            "Profiling-only applications did not compare against the mathematical reference or check output guards; correctness evidence is separate.",
            "Reported runtime register fields and exact ELF metadata are preserved separately; SDK descriptor reporting is not used as proof of register allocation change."]}
    by = {c["group"]: c for c in collections_}
    report["interface_diagnosis"] = {
        "status": "layer_clues_with_dependency_category_unresolved",
        "l2": {"evidence": by["l2_tagmap"]["matched_ratio_metrics"],
            "interpretation": "Substantial counted L2 hits coexist with misses; aggregate tagram shares do not locate a channel/set bottleneck. Tag stall/busy count ratio alone does not show L2 saturation."},
        "utcl1_and_external_credits": {"evidence": by["utcl1_credits"]["matched_ratio_metrics"],
            "raw_credit_events": {n: v for n, v in by["utcl1_credits"]["median_raw_csv_counter_values"].items() if "CREDIT_STALL" in n},
            "interpretation": "Read-credit stalls are zero in this pass, but this does not prove external memory has no latency cost. UTCL1 inflight and TA downstream stall events provide translation/downstream pressure clues, without wall-time attribution."},
        "read_service": {"evidence": by["latency"]["matched_ratio_metrics"],
            "interpretation": "EA request-cycle integral and TCP read latency show nonzero service residence; no clock/window calibration links these directly to nanoseconds or proves a single dominant block. UTCL1 LFIFO-not-resident events exist, but their elapsed-time share is unresolved."},
        "wave_and_lds": {"evidence": by["lds_issue"]["matched_ratio_metrics"],
            "raw_bank_conflict_count": by["lds_issue"]["median_raw_csv_counter_values"]["OCT8_SQ_LDS_BANK_CONFLICT_SUM"],
            "interpretation": "Cumulative dependency wait dominates instruction-issue wait. LDS-specific issue wait is a small part of wave life; bank-conflict events do not isolate which DS sites incurred them or establish total LDS dependency cost."},
        "next_evidence_needed": "Resolve which instructions create SQ_WAIT_ANY dependencies at prologue, MFMA operand loading and epilogue. ATT or another instruction-level dependency trace can distinguish VMEM-return, LDS-return and barrier waiting. Current aggregate counters do not justify selecting one kernel change as the confirmed root-cause fix."}
    output = HERE / "layer_analysis.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "summary": report["summary"],
        "metrics": {c["group"]: {n: m["statistics"]["median"] for n, m in c["matched_ratio_metrics"].items()} for c in collections_},
        "raw": {c["group"]: c["median_raw_csv_counter_values"] for c in collections_}, "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
