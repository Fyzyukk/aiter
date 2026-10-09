#!/usr/bin/env python3
"""CPU-only exact-binary ATT analysis of the current 9021 short/long K controls.

Read existing decoder output and the completed device audit. No compiler, HIP,
GPU tools, or kernel sources are invoked or changed. Only this directory's
att_analysis.json and att_mfma_gaps.csv are written.
"""
import collections
import csv
import hashlib
import json
from pathlib import Path
import re
import statistics
import struct

HERE = Path(__file__).resolve().parent
RESUME = HERE.parent
SFA = RESUME / "sfa_packed"
CONTRACT = ("gfx9 decoder event = [time, type, stall, duration, code_line]; "
            "successful MFMA issue = time + stall; duration includes stall; "
            "issue duration = duration - stall. All times are shader clocks. "
            "Wait duration is not an individual memory-return latency.")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def describe(values):
    values = sorted(values)
    if not values:
        return {"samples": 0}
    return {"samples": len(values), "mean": statistics.mean(values),
            "median": statistics.median(values),
            "p90": values[int(.9 * (len(values) - 1))],
            "min": values[0], "max": values[-1]}


def symbol_bytes(path, name):
    """Independent ELF64 symbol reader, matching audit_device.py hash scope."""
    data = Path(path).read_bytes()
    head = struct.unpack_from("<16sHHIQQQIHHHHHH", data)
    assert head[0][:6] == b"\x7fELF\x02\x01" and head[2] == 224
    sections = [struct.unpack_from("<IIQQQQIIQQ", data, head[6] + i * head[11])
                for i in range(head[12])]
    for section in sections:
        if section[1] not in (2, 11):
            continue
        strings_section = sections[section[6]]
        strings = data[strings_section[4]:strings_section[4] + strings_section[5]]
        for offset in range(section[4], section[4] + section[5], section[9]):
            n, info, _, si, value, size = struct.unpack_from("<IBBHQQ", data, offset)
            if n and strings[n:].split(b"\0", 1)[0].decode() == name:
                assert 0 < si < len(sections) and info & 15 == 2
                begin = sections[si][4] + value - sections[si][3]
                return value, data[begin:begin + size]
    raise ValueError(f"Missing kernel FUNC symbol: {name}")


def semantic(line, op):
    if op == "s_barrier":
        return "barrier"
    if op.startswith("s_waitcnt"):
        if line in (14, 107):
            return "kernarg_SMEM_wait"
        if line == 516:
            return "K0_matrix_and_SFA_VMEM_wait"
        if line in (540, 1168):
            return "SFB_VMEM_wait"
        if line == 1159:
            return "matrix_and_SFA_reload_VMEM_wait"
        if line in (592, 594, 738, 741, 744, 746):
            return "scale_LDS_wait"
        if line == 627:
            return "K0_matrix_and_SFB_LDS_wait"
        if line in (1173, 1175, 1178, 1187, 1273, 1282, 1291, 1298):
            return "matrix_LDS_wait"
        if line in (578, 548, 1192, 1198):
            return "matrix_VMEM_wait"
        if line in (580, 1171):
            return "scale_publish_and_other_LGKM_wait"
        if line == 751:
            return "empty_LGKM_wait_after_scale_reads"
        if line >= 1384:
            return "C_output_LDS_wait"
        return "other_wait"
    if op.startswith("buffer_load"):
        return "matrix_async_VMEM_issue" if " lds" in op else "scale_VMEM_issue"
    if op.startswith("s_load"):
        return "kernarg_SMEM_issue"
    if op.startswith("ds_read"):
        if line >= 1386:
            return "C_output_LDS_read"
        return "scale_LDS_read" if "u8" in op else "matrix_LDS_read"
    if op.startswith("ds_write"):
        return "C_output_LDS_write" if line >= 1306 else "scale_LDS_write"
    if op.startswith("buffer_store"):
        return "C_output_VMEM_store"
    if op.startswith("v_mfma_scale"):
        return "scaled_MFMA"
    if op.startswith("v_cvt_pk_bf16"):
        return "C_BF16_conversion"
    if op.startswith("s_nop"):
        return "explicit_nop"
    if op.startswith("s_endpgm"):
        return "endpgm_decoder_event"
    return "other_instructions"


def event(e, code, origin=0):
    row = code[e[4]]
    return {"code_line": e[4], "pc": hex(row[5]), "instruction": row[0],
            "time": e[0] - origin, "stall": e[2], "duration": e[3],
            "successful_issue": e[0] + e[2] - origin,
            "kind": semantic(e[4], row[0])}


def interval_breakdown(insts, code, begin, end):
    """Clip nonoverlapping exported event intervals; account for residual gaps."""
    result = collections.Counter()
    for e in insts:
        overlap = max(0, min(end, e[0] + e[3]) - max(begin, e[0]))
        if overlap:
            result[semantic(e[4], code[e[4]][0])] += overlap
    assert sum(result.values()) <= end - begin
    result["unattributed_decoder_gap"] += end - begin - sum(result.values())
    return dict(sorted(result.items()))


def wait_semantics(code, wave):
    rows = collections.defaultdict(list)
    for line, deps in wave["waitcnt"]:
        rows[line].append([{ "code_line": d[0], "pc": hex(code[d[0]][5]),
                             "instruction": code[d[0]][0],
                             "kind": semantic(d[0], code[d[0]][0])} for d in deps])
    return [{"code_line": line, "pc": hex(code[line][5]),
             "instruction": code[line][0], "kind": semantic(line, code[line][0]),
             "nonempty_dependency_records": len(deps),
             "distinct_dependency_sets": list({json.dumps(d, sort_keys=True): d
                                                 for d in deps}.values())}
            for line, deps in sorted(rows.items())]


def analyze_capture(k, reference, kernel):
    folder = HERE / f"current9021_k{k}_att"
    ui = next(folder.glob("ui_output_*"))
    application = json.loads((folder / "application.json").read_text())
    assert application["status"] == "passed"
    assert application["rows"][0]["kid"] == 9021
    assert application["rows"][0]["shape"] == [480, 7168, k]
    assert application["libraries"]["selected"]["sha256"] == reference["input_sha256"]
    assert digest(application["libraries"]["selected"]["path"]) == reference["input_sha256"]
    data = json.loads((ui / "code.json").read_text())
    assert data["header"] == "ISA, _, LineNumber, Source, Codeobj, Vaddr, Hit, Latency, Stall, Idle"
    code = {v[2]: v for v in data["code"]}
    assert len(code) == len(data["code"])
    objects = {v[4] for v in data["code"]}
    assert len(objects) == 1
    object_id = next(iter(objects))
    image = folder / f"current9021_k{k}_att_gfx950_code_object_id_{object_id}.out"
    assert digest(image) == reference["device_sha256"] == digest(reference["device"])
    start, instructions = symbol_bytes(image, kernel["name"])
    assert hashlib.sha256(instructions).hexdigest() == kernel["instruction_sha256"]
    assert len(instructions) == kernel["instruction_bytes"]
    isa = {}
    for line in (SFA / "device_audit_artifacts/baseline/kid9021.s").read_text().splitlines():
        match = re.match(r"^\s*(.*?)\s*// ([0-9A-Fa-f]+):", line)
        if match:
            isa[int(match[2], 16)] = re.sub(r"\s+", " ", match[1].strip())
    for v in data["code"]:
        if v[0].startswith(";"):
            assert kernel["name"] in v[0]
            continue
        assert start <= v[5] < start + len(instructions)
        assert re.sub(r"\s+", " ", v[0].strip()) == isa[v[5]]
    assert len(isa) == len(code) - 1 == 1498
    csv_stats = list(csv.DictReader(next(folder.glob("stats_ui_output_*.csv")).open()))
    assert len(csv_stats) == len(data["code"])
    for a, b in zip(csv_stats, data["code"]):
        assert int(a["CodeObj"]) == b[4] and int(a["Vaddr"]) == b[5]
        assert a["Instruction"] == b[0]
        assert [int(a[x]) for x in ["Hitcount", "Latency", "Stall", "Idle"]] == b[6:10]

    totals = collections.defaultdict(lambda: [0, 0, 0])
    waits = collections.defaultdict(list)
    all_windows = collections.defaultdict(list)
    all_breakdowns = collections.defaultdict(list)
    examples = collections.defaultdict(list)
    gap_csv = []
    per_wave = []
    wave_files = sorted(ui.glob("se*_sm*_sl*_wv*.json"))
    assert len(wave_files) == 4
    expected = (k // 128) * 4 * 4
    for file in wave_files:
        obj = json.loads(file.read_text())
        wave = obj["wave"]
        x = wave["instructions"]
        assert wave["cu"] == 0 and wave["slot"] == 0
        assert obj["num_insts"] == obj["num_stitched"] == len(x)
        assert all(len(e) == 5 and e[3] >= e[2] >= 0 for e in x)
        assert all(a[0] + a[3] <= b[0] for a, b in zip(x, x[1:]))
        positions = [i for i, e in enumerate(x) if code[e[4]][0].startswith("v_mfma_scale")]
        assert len(positions) == expected
        issues = [x[i][0] + x[i][2] for i in positions]
        for e in x:
            totals[e[4]][0] += 1
            totals[e[4]][1] += e[3]
            totals[e[4]][2] += e[2]
            if code[e[4]][0].startswith("s_waitcnt") or code[e[4]][0] == "s_barrier":
                waits[e[4]].append(event(e, code, wave["begin"]))
        def first(line):
            return next(e for e in x if e[4] == line)
        boundaries = [wave["begin"], first(14)[0] + first(14)[3],
                      first(107)[0] + first(107)[3], first(581)[0] + first(581)[3],
                      first(627)[0] + first(627)[3], issues[0]]
        phases = ["startup_kernargs", "K0_matrix_issue_and_scale_kernargs",
                  "scale_producer_K1_prefetch_and_publish_barrier",
                  "scale_and_K0_matrix_LDS_reads",
                  "runtime_tail_setup_accumulator_clear_and_K2_prefetch"]
        prologue = [{"phase": label, "clocks": end - begin,
                     "breakdown": interval_breakdown(x, code, begin, end)}
                    for label, begin, end in zip(phases, boundaries, boundaries[1:])]
        stores = [e for e in x if code[e[4]][0].startswith("buffer_store")]
        endpgm = next(e for e in x if code[e[4]][0] == "s_endpgm")
        first_store = stores[0][0] + stores[0][2]
        wave_record = {"wave": file.name, "cu": wave["cu"], "simd": wave["simd"],
                       "slot": wave["slot"], "begin": wave["begin"], "end": wave["end"],
                       "duration": wave["end"] - wave["begin"],
                       "scaled_MFMA": len(positions),
                       "prologue_to_first_MFMA": issues[0] - wave["begin"],
                       "first_to_last_MFMA": issues[-1] - issues[0],
                       "last_MFMA_to_wave_end": wave["end"] - issues[-1],
                       "prologue_phases": prologue,
                       "tile0_to_tile1_MFMA_issue": issues[16] - issues[0],
                       "tile1_to_tile2_MFMA_issue": issues[32] - issues[16],
                       "final_tile_MFMA_issue_span": issues[-1] - issues[-16],
                       "last_MFMA_to_first_global_store": first_store - issues[-1],
                       "global_store_begin_to_endpgm": endpgm[0] - first_store,
                       "endpgm_attempt_to_wave_end": wave["end"] - endpgm[0],
                       "whole_wave_breakdown": interval_breakdown(x, code, wave["begin"], wave["end"]),
                       "wait_dependencies": wait_semantics(code, wave),
                       "timeline_sum_minus_wave_duration": sum(t[1] for t in wave["timeline"]) - (wave["end"] - wave["begin"])}
        per_wave.append(wave_record)
        for ordinal in range(len(positions) - 1):
            p, q = positions[ordinal:ordinal + 2]
            a, b = x[p], x[q]
            begin, end = issues[ordinal:ordinal + 2]
            gap = end - begin
            tile, index = divmod(ordinal, 16)
            reload = any(e[4] == 1172 for e in x[p + 1:q])
            if tile == 0:
                scope = "startup_tile"
            elif tile == k // 128 - 1:
                scope = "final_tile_with_output"
            elif tile >= k // 128 - 2:
                scope = "penultimate_tile"
            else:
                scope = "interior_tiles"
            name = f"{scope}/gap_{index:02d}_{(index + 1) % 16:02d}"
            if index == 15:
                name += "/scale_panel_reload" if reload else "/ordinary"
            breakdown = interval_breakdown(x[p:q + 1], code, begin, end)
            all_windows[name].append(gap)
            all_breakdowns[name].append(breakdown)
            if index in (3, 15) or scope == "final_tile_with_output":
                examples[name].append((gap, {"wave": file.name, "k_tile": tile,
                                            "gap": gap, "breakdown": breakdown,
                                            "instructions": [event(e, code, begin) for e in x[p:q + 1]]}))
            gap_csv.append([k, file.name, tile, index, scope, hex(code[a[4]][5]),
                            hex(code[b[4]][5]), gap, reload, json.dumps(breakdown, sort_keys=True)])
    assert all(totals[v[2]] == v[6:9] for v in data["code"])
    def summary(field):
        return describe([w[field] for w in per_wave])
    windows = {}
    for name, vals in sorted(all_windows.items()):
        keys = set().union(*all_breakdowns[name])
        windows[name] = {"gap": describe(vals), "event_interval_mean_clocks":
                         {key: statistics.mean(d.get(key, 0) for d in all_breakdowns[name])
                          for key in sorted(keys)}}
    barrier_groups = []
    for line in [581, 752, 1172, 1299, 1385]:
        grouped = collections.defaultdict(list)
        for file in wave_files:
            w = json.loads(file.read_text())["wave"]
            for occurrence, e in enumerate([e for e in w["instructions"] if e[4] == line]):
                grouped[occurrence].append({"wave": file.name, "attempt": e[0],
                                            "release": e[0] + e[3], "duration": e[3]})
        for occurrence, vals in grouped.items():
            barrier_groups.append({"pc": hex(code[line][5]), "occurrence": occurrence,
                                   "wave_count": len(vals), "attempt_spread": max(v["attempt"] for v in vals) - min(v["attempt"] for v in vals),
                                   "release_spread": max(v["release"] for v in vals) - min(v["release"] for v in vals),
                                   "waves": vals})
    report = {"k": k, "capture": str(folder), "ui": str(ui), "waves": 4,
              "grid_workgroups": 224, "expected_MFMA_per_wave": expected,
              "identity": {"device_sha256": digest(image), "kernel_instruction_sha256": kernel["instruction_sha256"],
                           "kernel_instruction_bytes": len(instructions), "PC_instruction_matches": 1498,
                           "code_json_sha256": digest(ui / "code.json"),
                           "application_sha256": digest(folder / "application.json"),
                           "wave_hit_duration_stall_matches_all_code_rows": True,
                           "stats_csv_matches_code_json": True},
              "summary": {field: summary(field) for field in ["duration", "prologue_to_first_MFMA",
                          "first_to_last_MFMA", "last_MFMA_to_wave_end", "tile0_to_tile1_MFMA_issue",
                          "tile1_to_tile2_MFMA_issue", "final_tile_MFMA_issue_span",
                          "last_MFMA_to_first_global_store", "endpgm_attempt_to_wave_end"]},
              "phase_share_mean_percent": {field: statistics.mean(100 * w[field] / w["duration"] for w in per_wave)
                                           for field in ["prologue_to_first_MFMA", "first_to_last_MFMA", "last_MFMA_to_wave_end"]},
              "prologue_phase_means": {phase: statistics.mean(next(v["clocks"] for v in w["prologue_phases"] if v["phase"] == phase) for w in per_wave) for phase in phases},
              "wait_barrier_by_PC": [{"code_line": line, "pc": hex(code[line][5]), "instruction": code[line][0],
                                       "kind": semantic(line, code[line][0]), "duration": describe([v["duration"] for v in vals]),
                                       "duration_sum": sum(v["duration"] for v in vals),
                                       "stall_sum": sum(v["stall"] for v in vals)} for line, vals in sorted(waits.items())],
              "successful_MFMA_issue_windows": windows,
              "representative_windows": {name: min(vals, key=lambda v: abs(v[0] - statistics.median(all_windows[name])))[1]
                                         for name, vals in sorted(examples.items())},
              "barrier_occurrences": barrier_groups, "per_wave": per_wave}
    return report, gap_csv


def main():
    audit = json.loads((SFA / "device_audit.json").read_text())
    reference = next(x for x in audit["libraries"] if x["label"] == "baseline")
    kernel = next(x for x in reference["kernels"] if x["kid"] == 9021)
    captures, rows = [], []
    for k in [384, 16384]:
        capture, gap_rows = analyze_capture(k, reference, kernel)
        captures.append(capture)
        rows += gap_rows
    report = {"status": "passed", "cpu_only": True, "gpu_executed": False, "build_executed": False,
              "timing_contract": CONTRACT,
              "scope": "One capture per K, same unchanged selected 9021 binary and 224-WG grid. Only four full waves from SE0/CU0, one slot on each SIMD, have stitched instruction timelines. Waves and repeated tile windows are correlated, not independent performance repetitions. Occupancy includes wave activity on other CUs without their instruction timelines. No all-CU or no-profiler performance conclusion follows.",
              "decoder_endpoint_note": "Exported timeline/event end can exceed wave end by one 4-clock decoder quantum on three waves per capture; analyses clip intervals to wave end and use wave begin/end for durations.",
              "dependency_note": "waitcnt dependency records list nonempty waits only and can omit empty dynamic occurrences. Sets below establish instruction queue membership, not individual memory-return timestamps. Partial LGKM waits can have scale reads outstanding while gating preceding matrix loads.",
              "analysis_script_sha256": digest(__file__), "device_audit_sha256": digest(SFA / "device_audit.json"),
              "captures": captures,
              "findings": [
                  "Short K: first-MFMA prologue occupies about half of CU0 wave duration. Scalar kernarg waits at 0x1f58/0x2150, scale producer serialization, scale-publish wave imbalance, runtime setup/accumulator clear, and K2 prefetch all contribute; SFA LDS packing alone does not explain it.",
                  "Short K scale producer: only SIMD0 executes SFA vector16 VMEM 0x29e4, then vmcnt(0) 0x29f8 and LDS write 0x29fc; SFB VMEM 0x2a64 begins afterwards, then vmcnt(0) 0x2a74 and LDS write 0x2a78. Other waves arrive earlier at barrier 0x2b5c. Its releases align within four clocks. The first VMEM wait depends on eight K0 matrix async loads plus SFA; its duration must not be attributed to SFA alone.",
                  "Long K: prologue share drops to about 3.5%; normal interior gaps 15→next tile 0 contain next matrix LDS reads, runtime ring bookkeeping, and eight async VMEM issue sites. Separate reload boundaries at tiles 30/62/94 add SFA VMEM wait/publish/barrier. These are different interfaces and must be measured separately.",
                  "Interior gap 3→4 includes next-scale LDS waits/packing, ring-stage bookkeeping, and barrier 0x2f18 before a useful MFMA whose current operands are already resident. The trace proves local issue delay at these PCs but does not prove a particular scheduling rewrite is faster or globally safe.",
                  "Final K tile interleaves BF16 conversion, C LDS writes and explicit nops with remaining MFMA. Its wider gaps and post-last-MFMA output/store/endpgm interval must not be labeled steady-state MFMA or scale stall."
              ]}
    (HERE / "att_analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    with (HERE / "att_mfma_gaps.csv").open("w", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(["K", "wave", "K_tile", "MFMA_index", "phase", "PC", "next_PC", "successful_issue_gap_clocks", "scale_panel_reload", "event_interval_breakdown"])
        writer.writerows(rows)
    for capture in captures:
        print(json.dumps({"K": capture["k"], "summary": capture["summary"],
                          "phase_share_percent": capture["phase_share_mean_percent"],
                          "prologue_phase_means": capture["prologue_phase_means"]}, indent=2))


if __name__ == "__main__":
    main()
