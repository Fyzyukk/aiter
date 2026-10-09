#!/usr/bin/env python3
"""CPU-only candidate/baseline K384 ATT comparison; never build or use GPU.

Read the existing captures and exact-device audit. Write only the new
scale_issue_publish_att_analysis.json beside this script. All event times are
shader clocks, with successful MFMA issue = time + stall.
"""
import collections
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import statistics

HERE = Path(__file__).resolve().parent
RESUME = HERE.parent
spec = importlib.util.spec_from_file_location("baseline_att_helpers", HERE / "analyze_att.py")
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)

PCS = {
    "baseline": {"first_kernarg_wait": 0x1f58, "scale_kernarg_wait": 0x2150,
                 "SFA_issue": 0x29e4, "SFA_wait": 0x29f8, "SFA_publish": 0x29fc,
                 "SFB_issue": 0x2a64, "SFB_wait": 0x2a74, "SFB_publish": 0x2a78,
                 "publish_LGKM_wait": 0x2b58, "publish_barrier": 0x2b5c,
                 "scale_LDS_wait1": 0x2bb0, "scale_LDS_wait0": 0x2bbc,
                 "K0_matrix_SFB_LDS_wait": 0x2ca4},
    "candidate": {"first_kernarg_wait": 0x1f50, "scale_kernarg_wait": 0x2148,
                  "SFA_issue": 0x29ec, "SFB_issue": 0x2a48,
                  "both_scale_and_K0_VMEM_wait": 0x2a50, "SFA_wait": 0x2a70,
                  "SFA_publish": 0x2a74, "SFB_publish": 0x2a8c,
                  "publish_LGKM_wait": 0x2b74, "publish_barrier": 0x2b78,
                  "scale_LDS_wait1": 0x2bcc, "scale_LDS_wait0": 0x2bd8,
                  "K0_matrix_SFB_LDS_wait": 0x2cc0},
}


def normalize(text):
    return re.sub(r"\s+", " ", text.strip())


def capture(label, folder, reference, kernel, isa_path):
    ui = next(folder.glob("ui_output_*"))
    app = json.loads((folder / "application.json").read_text())
    assert app["status"] == "passed" and app["rows"][0]["shape"] == [480, 7168, 384]
    assert app["rows"][0]["kid"] == 9021 and app["rows"][0]["profile_iterations"] == 11
    assert app["rows"][0]["profile_rotation"] == 1
    library_label = "selected" if label == "baseline" else "candidate"
    library = app["libraries"][library_label]
    assert library["sha256"] == reference["input_sha256"] == helpers.digest(library["path"])
    data = json.loads((ui / "code.json").read_text())
    assert data["header"] == "ISA, _, LineNumber, Source, Codeobj, Vaddr, Hit, Latency, Stall, Idle"
    code = {r[2]: r for r in data["code"]}
    assert len(code) == len(data["code"])
    ids = {r[4] for r in code.values()}
    assert len(ids) == 1
    image = next(folder.glob(f"*_code_object_id_{next(iter(ids))}.out"))
    assert helpers.digest(image) == reference["device_sha256"] == helpers.digest(reference["device"])
    start, func = helpers.symbol_bytes(image, kernel["name"])
    assert len(func) == kernel["instruction_bytes"]
    assert hashlib.sha256(func).hexdigest() == kernel["instruction_sha256"]
    isa = {}
    for line in isa_path.read_text().splitlines():
        match = re.match(r"^\s*(.*?)\s*// ([0-9A-Fa-f]+):", line)
        if match:
            isa[int(match[2], 16)] = normalize(match[1])
    for row in code.values():
        if row[0].startswith(";"):
            assert kernel["name"] in row[0]
        else:
            assert start <= row[5] < start + len(func)
            assert normalize(row[0]) == isa[row[5]]
    assert len(isa) == len(code) - 1
    csv_rows = list(csv.DictReader(next(folder.glob("stats_ui_output_*.csv")).open()))
    assert len(csv_rows) == len(data["code"])
    for a, b in zip(csv_rows, data["code"]):
        assert int(a["CodeObj"]) == b[4] and int(a["Vaddr"]) == b[5]
        assert a["Instruction"] == b[0]
        assert [int(a[k]) for k in ["Hitcount", "Latency", "Stall", "Idle"]] == b[6:10]
    files = sorted(ui.glob("se*_sm*_sl*_wv*.json"))
    assert len(files) == 4
    aggregate = collections.defaultdict(lambda: [0, 0, 0])
    role_events = collections.defaultdict(list)
    waves = []
    all_dependency_sets = collections.defaultdict(dict)
    idle = 0
    for file in files:
        obj = json.loads(file.read_text())
        w = obj["wave"]
        insts = w["instructions"]
        assert w["cu"] == 0 and w["slot"] == 0
        assert obj["num_insts"] == obj["num_stitched"] == len(insts)
        assert all(len(e) == 5 and e[3] >= e[2] >= 0 for e in insts)
        assert all(a[0] + a[3] <= b[0] for a, b in zip(insts, insts[1:]))
        timeline_delta = sum(e[1] for e in w["timeline"]) - (w["end"] - w["begin"])
        assert timeline_delta in (0, 4)
        idle += insts[0][0] - w["begin"]
        idle += sum(b[0] - a[0] - a[3] for a, b in zip(insts, insts[1:]))
        bypc = collections.defaultdict(list)
        for e in insts:
            bypc[code[e[4]][5]].append(e)
            total = aggregate[e[4]]
            total[0] += 1; total[1] += e[3]; total[2] += e[2]
        mfma = [e for e in insts if code[e[4]][0].startswith("v_mfma_scale")]
        assert len(mfma) == 48
        issue = lambda e: e[0] + e[2]
        def end(role):
            values = bypc[PCS[label][role]]
            assert len(values) == 1
            return values[0][0] + values[0][3]
        boundaries = [w["begin"], end("first_kernarg_wait"), end("scale_kernarg_wait"),
                      end("publish_barrier"), end("K0_matrix_SFB_LDS_wait"), issue(mfma[0])]
        phase_labels = ["startup_kernargs", "K0_matrix_issue_and_scale_kernargs",
                        "scale_producer_K1_prefetch_and_publish_barrier", "scale_and_K0_matrix_LDS_reads",
                        "runtime_setup_clear_K2_prefetch"]
        record = {"wave": file.name, "simd": w["simd"], "begin": w["begin"], "end": w["end"],
                  "duration": w["end"] - w["begin"], "scaled_MFMA": 48,
                  "prologue_to_first_MFMA": issue(mfma[0]) - w["begin"],
                  "first_to_last_MFMA": issue(mfma[-1]) - issue(mfma[0]),
                  "last_MFMA_to_wave_end": w["end"] - issue(mfma[-1]),
                  "prologue_share_percent": 100 * (issue(mfma[0]) - w["begin"]) / (w["end"] - w["begin"]),
                  "phase_clocks": dict(zip(phase_labels, [b - a for a, b in zip(boundaries, boundaries[1:])])),
                  "timeline_sum_minus_wave_duration": timeline_delta,
                  "role_events": {}}
        for role, pc in PCS[label].items():
            vals = []
            for e in bypc[pc]:
                value = {"wave": file.name, "simd": w["simd"], "pc": hex(pc),
                         "code_line": e[4], "instruction": code[e[4]][0], "time": e[0],
                         "time_from_wave_begin": e[0] - w["begin"], "stall": e[2], "duration": e[3],
                         "successful_issue": issue(e), "event_end": e[0] + e[3]}
                vals.append(value)
                role_events[role].append(value)
            record["role_events"][role] = vals
        if bypc[PCS[label]["SFA_issue"]]:
            sfa, sfb = bypc[PCS[label]["SFA_issue"]][0], bypc[PCS[label]["SFB_issue"]][0]
            pubs = bypc[PCS[label]["SFB_publish"]]
            assert len(pubs) == 1
            record["producer_window"] = {"SFA_to_SFB_successful_issue": issue(sfb) - issue(sfa),
                                         "SFA_successful_issue_to_both_publish_event_end": pubs[0][0] + pubs[0][3] - issue(sfa),
                                         "SFA_successful_issue_to_publish_barrier_release": end("publish_barrier") - issue(sfa)}
        for line, deps in w["waitcnt"]:
            pc = code[line][5]
            if pc in PCS[label].values():
                dep = [{"pc": hex(code[d[0]][5]), "instruction": code[d[0]][0]} for d in deps]
                all_dependency_sets[hex(pc)][json.dumps(dep, sort_keys=True)] = dep
        waves.append(record)
    assert all(aggregate[r[2]] == r[6:9] for r in code.values())
    assert sum(r[9] for r in code.values()) == idle
    summary = {field: helpers.describe([w[field] for w in waves])
               for field in ["duration", "prologue_to_first_MFMA", "first_to_last_MFMA", "last_MFMA_to_wave_end", "prologue_share_percent"]}
    roles = {role: {"pc": hex(PCS[label][role]), "events": vals,
                    "duration": helpers.describe([v["duration"] for v in vals]),
                    "duration_sum": sum(v["duration"] for v in vals)} for role, vals in role_events.items()}
    barrier = role_events["publish_barrier"]
    assert len(barrier) == 4
    return {"label": label, "capture": str(folder), "ui": str(ui),
            "identity": {"library_sha256": helpers.digest(library["path"]), "device_sha256": helpers.digest(image),
                         "kernel_instruction_sha256": kernel["instruction_sha256"], "kernel_instruction_bytes": len(func),
                         "PC_instruction_matches": len(isa), "code_json_sha256": helpers.digest(ui / "code.json"),
                         "wave_hit_duration_stall_aggregates_match_all_code_rows": True,
                         "stats_csv_matches_code_json": True, "code_idle_equals_wave_lead_and_interevent_gaps": True,
                         "timeline_endpoints_match_within_one_4_clock_quantum": True,
                         "complete_waves": 4, "scaled_MFMA_each_wave": 48},
            "summary": summary,
            "phase_mean_clocks": {phase: statistics.mean(w["phase_clocks"][phase] for w in waves) for phase in waves[0]["phase_clocks"]},
            "roles": roles, "publish_barrier": {"attempt_spread": max(v["time"] for v in barrier) - min(v["time"] for v in barrier),
                                                "release_spread": max(v["event_end"] for v in barrier) - min(v["event_end"] for v in barrier),
                                                "duration_sum": sum(v["duration"] for v in barrier), "waves": barrier},
            "wait_dependency_sets": {pc: list(vals.values()) for pc, vals in all_dependency_sets.items()},
            "waves": waves}


def main():
    old_path = HERE / "att_analysis.json"
    old_hash = helpers.digest(old_path)
    old = next(c for c in json.loads(old_path.read_text())["captures"] if c["k"] == 384)
    captures = []
    for label, package, folder in [
            ("baseline", RESUME / "sfa_packed", HERE / "current9021_k384_att"),
            ("candidate", RESUME / "scale_issue_publish", HERE / "scale_issue_publish_k384_att")]:
        audit = json.loads((package / "device_audit.json").read_text())
        ref = next(v for v in audit["libraries"] if v["label"] == label)
        kernel = next(k for k in ref["kernels"] if k["kid"] == 9021)
        captures.append(capture(label, folder, ref, kernel, package / f"device_audit_artifacts/{label}/kid9021.s"))
    baseline, candidate = captures
    assert baseline["identity"]["code_json_sha256"] == old["identity"]["code_json_sha256"]
    for field in ["duration", "prologue_to_first_MFMA", "first_to_last_MFMA", "last_MFMA_to_wave_end"]:
        assert baseline["summary"][field] == old["summary"][field]
    delta = {field: candidate["summary"][field]["mean"] - baseline["summary"][field]["mean"]
             for field in baseline["summary"]}
    phase_delta = {phase: candidate["phase_mean_clocks"][phase] - baseline["phase_mean_clocks"][phase]
                   for phase in baseline["phase_mean_clocks"]}
    before = sum(baseline["roles"][k]["duration_sum"] for k in ["SFA_wait", "SFB_wait"])
    after = sum(candidate["roles"][k]["duration_sum"] for k in ["both_scale_and_K0_VMEM_wait", "SFA_wait"])
    report = {"status": "passed", "cpu_only": True, "gpu_executed": False, "build_executed": False,
              "analysis_script_sha256": helpers.digest(__file__), "baseline_analysis_sha256": old_hash,
              "timing_contract": helpers.CONTRACT,
              "scope": "One baseline and one candidate capture, one stitched CU0 workgroup with four Wave64 timelines per capture, same K384 shape/224-WG grid/SE0/SIMD15 settings. Waves are correlated; capture timing differs and includes profiler perturbation. Absolute local-clock differences do not establish all-CU latency, throughput or a benchmark speedup.",
              "candidate_minus_baseline_mean_shader_clocks": delta,
              "candidate_minus_baseline_phase_mean_shader_clocks": phase_delta,
              "serial_scale_VMEM_wait_event_duration_sum": {"baseline": before, "candidate": after, "candidate_minus_baseline": after - before,
                  "interpretation": "Baseline waits depend first on K0 matrix VMEM plus SFA, then SFB. Candidate wait0x2a50 lists K0 matrix+SFA+SFB; wait0x2a70 is empty and4 clocks. These are queue-wait events, not individual load return latencies."},
              "publish_barrier_duration_sum": {"baseline": baseline["publish_barrier"]["duration_sum"],
                  "candidate": candidate["publish_barrier"]["duration_sum"],
                  "candidate_minus_baseline": candidate["publish_barrier"]["duration_sum"] - baseline["publish_barrier"]["duration_sum"]},
              "captures": captures,
              "findings": [
                  "The candidate's traced producer issues both SFA and SFB VMEM before either SFA/SFB LDS publish. Baseline issues SFB after SFA VMEM wait and publish. This confirms the intended issue-order mechanism in the executed full-vector path.",
                  "Candidate shared VMEM wait and earlier publish-barrier convergence are smaller in this finite capture; first-MFMA prologue also shortens. Separate leading SMEM and post-prologue changes must remain visible rather than assigning the entire total-wave difference to the scale change.",
                  "Candidate nonproducer waves export a ds_write_b8 event at0x2a8c without an SFB VMEM event. ATT instruction events do not reveal active-lane memory transactions; no transfer-volume, guard correctness or active-lane count is inferred from those events. Guards and correctness require source review and separate validated tests.",
                  "Compare each binary with its own ISA PCs: candidate0x2a74 is ds_write_b128, whereas baseline0x2a74 is the SFB vmcnt wait. Reusing baseline PC meanings for candidate would give an incorrect conclusion."
              ]}
    assert helpers.digest(old_path) == old_hash
    (HERE / "scale_issue_publish_att_analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "candidate_minus_baseline_mean_shader_clocks": delta,
                      "phase_deltas": phase_delta, "scale_wait": report["serial_scale_VMEM_wait_event_duration_sum"],
                      "barrier": report["publish_barrier_duration_sum"]}, indent=2))


if __name__ == "__main__":
    main()
