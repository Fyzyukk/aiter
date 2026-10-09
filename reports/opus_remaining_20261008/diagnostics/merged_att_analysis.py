#!/usr/bin/env python3
"""Read current Oct8 ATT evidence for 9020/9022/9030, using only CPU tools.

Each run refreshes this new analysis JSON/Markdown from completed clean claims.
It never runs HIP, a compiler, a profiler, or modifies kernel/history files.
"""
import collections
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import statistics
import struct
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OLD = ROOT / "reports/opus_bound_analysis_20261007"
FORMAL = ROOT / "reports/opus_resume_20261008/formal_selected"
PARENTS = {9020, 9022, 9030}
CONTRACT = ("gfx9 event [time,type,stall,duration,code_line]: successful issue "
            "= time+stall; duration includes stall. All event times are shader "
            "clocks. Wait dependencies identify queue members, not individual "
            "memory-return latency. Wave/event shares are not full-kernel wall-time shares.")


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    require("torch" not in sys.modules, "CPU analysis imported torch")
    return result


COMMON = module("remaining_cpu_elf", FORMAL / "smoke_common.py")
CLAIM_HELPER = module("remaining_cpu_claim", OLD / "run_when_idle.py")


def stats(values):
    values = sorted(values)
    if not values:
        return {"samples": 0}
    return {"samples": len(values), "min": values[0], "median": statistics.median(values),
            "mean": statistics.mean(values), "p90": values[int(.9 * (len(values) - 1))],
            "max": values[-1]}


def normalized(op):
    return re.sub(r"\s+", " ", op.strip())


def symbol_address(elf, name):
    for row in elf.rows:
        if row[1] not in (2, 11):
            continue
        strings = elf.bytes(elf.rows[row[6]])
        for off in range(row[4], row[4] + row[5], row[9]):
            n, info, _, si, value, size = struct.unpack_from("<IBBHQQ", elf.data, off)
            if n and elf.string(strings, n) == name and si:
                require(info & 15 == 2, "Selected symbol is not FUNC")
                return value, size
    raise ValueError("Missing FUNC address")


def clean_claim(name, app, command, queue, claims):
    starts = [c for c in claims if c.get("event") == "start"
              and c.get("command", {}).get("name") == name and c["time"] <= app["started"]]
    require(starts, "No command start before application")
    start = max(starts, key=lambda c: c["time"])
    ends = [c for c in claims if c.get("event") == "end" and c.get("name") == name
            and c["time"] >= app["finished"]]
    require(ends, "Command has not completed its clean claim")
    end = min(ends, key=lambda c: c["time"])
    epoch = [c for c in claims if start["time"] <= c["time"] <= end["time"]]
    require(start["command"] == command, "Executed command differs from current queue")
    require(end["returncode"] == 0 and end["contamination"] is False, "Unclean command end")
    require(not any(c.get("event") in ("external_work_started", "owner_identity_unresolved", "claimed")
                    for c in epoch), "Claim changed or external work appeared during command")
    claim = max((c for c in claims if c.get("event") == "claimed" and c["time"] <= start["time"]),
                key=lambda c: c["time"])
    owners = [c for c in epoch if c.get("event") == "owner_identity" and c.get("name") == name]
    require(len(owners) == 1, "Expected one same-mm owner marker")
    owner = owners[0]
    require(owner["new_host_pids"] == [owner["host_pid"]]
            and owner["host_pid"] not in owner["baseline_host_pids"]
            and owner["launcher_sha256"] == sha(OLD / "owned_python_launch.py"), "Owner marker differs")
    monitors = [c for c in epoch if c.get("event") == "monitor" and c["time"] >= owner["time"]]
    require(monitors and all(c["child_pid"] == owner["inner_pid"] for c in monitors), "Monitor PID differs")
    require(all(all(p["pid"] == owner["host_pid"] for p in c["processes"]) for c in monitors),
            "Monitor contains a process other than the recorded owner")
    gpu = app["gpu"]
    require(gpu["pci_bdf"] == gpu["expected_pci_bdf"] == claim["gpu"]["bdf"]
            and gpu["HIP_VISIBLE_DEVICES"] == str(claim["gpu"]["hip_index"]), "Physical GPU mismatch")
    fingerprint = CLAIM_HELPER.command_fingerprint(command, {**queue.get("env", {}), **command.get("env", {})})
    require(fingerprint == start["fingerprint"], "Source/plan/binary command fingerprint differs")
    return {"claim": claim, "start": start, "end": end, "owner_identity": owner,
            "monitor_count": len(monitors), "fingerprint_recomputed_equal": True}


def base_kind(op, output_pc):
    if op == "s_barrier":
        return "barrier"
    if op.startswith("s_waitcnt"):
        return "waitcnt"
    if op.startswith("s_load"):
        return "kernarg_SMEM_issue"
    if op.startswith("buffer_load"):
        if " lds" in op:
            return "matrix_async_VMEM_issue"
        return "SFB_scale_VMEM_issue" if "ubyte" in op else "SFA_scale_VMEM_issue"
    if op.startswith("ds_read"):
        return "scale_LDS_read" if "u8" in op else "matrix_or_output_LDS_read"
    if op.startswith("ds_write"):
        return "LDS_write"
    if op.startswith("v_mfma_scale"):
        return "scaled_MFMA"
    if op.startswith("v_cvt_pk_bf16"):
        return "BF16_conversion"
    if op.startswith("buffer_store"):
        return "C_VMEM_store"
    if op == "s_endpgm":
        return "endpgm_decoder_duration"
    if op.startswith("s_nop"):
        return "explicit_nop"
    return "other_instructions"


def make_kinds(code, waves):
    output_pc = min(r[5] for r in code.values() if r[0].startswith("v_cvt_pk_bf16"))
    kinds = {}
    deps = collections.defaultdict(set)
    for w in waves:
        for line, records in w["waitcnt"]:
            for d in records:
                deps[line].add(d[0])
    for line, row in code.items():
        op, pc = row[0], row[5]
        kind = base_kind(op, output_pc)
        if op.startswith("ds_write"):
            kind = "C_LDS_write" if pc >= output_pc else "scale_LDS_publish"
        elif op.startswith("ds_read") and pc >= output_pc:
            kind = "C_LDS_read"
        elif op.startswith("s_waitcnt"):
            member_kinds = {base_kind(code[d][0], output_pc) for d in deps.get(line, ())}
            scale = bool(member_kinds & {"SFA_scale_VMEM_issue", "SFB_scale_VMEM_issue"})
            matrix = "matrix_async_VMEM_issue" in member_kinds
            smem = "kernarg_SMEM_issue" in member_kinds
            if pc >= output_pc:
                kind = "C_output_waitcnt"
            elif matrix and scale:
                kind = "matrix_and_scale_VMEM_waitcnt"
            elif scale:
                kind = "scale_VMEM_waitcnt"
            elif matrix:
                kind = "matrix_VMEM_waitcnt"
            elif smem:
                kind = "kernarg_SMEM_waitcnt"
            elif "vmcnt" in op:
                kind = "VMEM_waitcnt_without_exported_members"
            else:
                kind = "LDS_or_other_LGKM_waitcnt"
        kinds[line] = kind
    return kinds, deps, output_pc


def event(e, code, kinds, origin):
    r = code[e[4]]
    return {"code_line": e[4], "pc": hex(r[5]), "instruction": r[0], "kind": kinds[e[4]],
            "time": e[0] - origin, "stall": e[2], "duration": e[3],
            "successful_issue": e[0] + e[2] - origin}


def breakdown(x, kinds, begin, end):
    result = collections.Counter()
    for index, e in enumerate(x):
        # The decoder can extend s_mov m0 duration four clocks into the next
        # s_nop. Retain original aggregate durations; only interval partition
        # ownership is capped at the next event's time to avoid double count.
        event_end = min(e[0] + e[3], x[index + 1][0]) if index + 1 < len(x) else e[0] + e[3]
        overlap = max(0, min(end, event_end) - max(begin, e[0]))
        if overlap:
            result[kinds[e[4]]] += overlap
    require(sum(result.values()) <= end - begin, "Overlapping instruction intervals")
    result["unattributed_decoder_gap"] += end - begin - sum(result.values())
    return dict(sorted(result.items()))


def capture(index, target, plan, queue, claims, reference, linked_shas, suffix="", rejected_attempts=None):
    name = f"target{index}_kid{target['kid']}_att" + suffix
    folder = HERE / name
    app = read(folder / "application.json")
    require(app["status"] == "passed" and not app.get("contamination"), "Application incomplete or contaminated")
    row = app["rows"][0]
    require(len(app["rows"]) == 1 and row["target_index"] == index
            and row["kid"] == target["kid"] and row["shape"] == target["shape"], "Target differs")
    require(app["profiling_only"] and row["profile_label"] == "official", "Not official profiler execution")
    require(app["official_binary_sha256"] == plan["official_binary_sha256"]
            and row["actual_official_module"]["sha256"] == plan["official_binary_sha256"]
            and row["actual_official_module"]["path"] == plan["official_binary"], "Module record differs")
    command = next(c for c in queue["commands"] if c["name"] == name)
    claim = clean_claim(name, app, command, queue, claims)
    uis = list(folder.glob("ui_output*"))
    require(len(uis) == 1 and (uis[0] / "code.json").is_file(), "Missing single decoded ATT capture")
    ui = uis[0]
    data = read(ui / "code.json")
    require(data["header"] == "ISA, _, LineNumber, Source, Codeobj, Vaddr, Hit, Latency, Stall, Idle", "Decoder schema differs")
    code = {r[2]: r for r in data["code"]}
    require(len(code) == len(data["code"]), "Duplicate code line")
    objects = {r[4] for r in code.values()}
    require(len(objects) == 1, "Multiple code objects in one selected function")
    oid = next(iter(objects))
    image = folder / f"{name}_gfx950_code_object_id_{oid}.out"
    require(sha(image) in linked_shas, "Captured CO SHA not present in current linked module audit")
    elf = COMMON.Elf(image.read_bytes())
    require(elf.machine == 224, "Captured image is not AMDGPU ELF")
    symbols = elf.symbols()
    metadata = next(k for k in elf.metadata()["amdhsa.kernels"] if k[".name"] == target["symbol"])
    ref = reference[target["symbol"]]
    func = symbols[target["symbol"]]
    descriptor = bytearray(symbols[metadata[".symbol"]]["bytes"])
    require(len(descriptor) == 64, "Descriptor length differs")
    descriptor[16:24] = b"\0" * 8
    require(metadata == ref["metadata"] and func["type"] == 2
            and func["size"] == ref["instruction_bytes"]
            and hashlib.sha256(func["bytes"]).hexdigest() == ref["instruction_sha256"] == target["instruction_sha256"]
            and hashlib.sha256(descriptor).hexdigest() == ref["descriptor_normalized_sha256"], "Current FUNC/metadata/descriptor mismatch")
    begin, size = symbol_address(elf, target["symbol"])
    disasm = subprocess.check_output(["/opt/rocm/llvm/bin/llvm-objdump", "-d", "--mcpu=gfx950",
                                      "--disassemble-symbols=" + target["symbol"], str(image)], text=True)
    isa = {}
    for line in disasm.splitlines():
        match = re.match(r"^\s*(.*?)\s*// ([0-9a-fA-F]+):", line)
        if match:
            isa[int(match[2], 16)] = normalized(match[1])
    for r in code.values():
        if r[0].startswith(";"):
            require(target["symbol"] in r[0], "Decoder label differs")
        else:
            require(begin <= r[5] < begin + size and normalized(r[0]) == isa.get(r[5]), "Decoder PC/instruction differs from captured FUNC")
    require(len(isa) == len(code) - 1, "Decoder function coverage differs")
    csv_path = next(folder.glob("stats_ui_output*.csv"))
    csv_rows = list(csv.DictReader(csv_path.open()))
    require(len(csv_rows) == len(data["code"]), "Stats CSV coverage differs")
    for a, b in zip(csv_rows, data["code"]):
        require(a["Instruction"] == b[0] and int(a["CodeObj"]) == b[4] and int(a["Vaddr"]) == b[5]
                and [int(a[k]) for k in ("Hitcount", "Latency", "Stall", "Idle")] == b[6:10], "Stats CSV does not match code.json")
    wave_paths = sorted(ui.glob("se*_sm*_sl*_wv*.json"))
    objs = [read(p) for p in wave_paths]
    waves = [o["wave"] for o in objs]
    expected_waves = metadata[".max_flat_workgroup_size"] // metadata[".wavefront_size"]
    if target["kid"] == 9022:
        per_tile = 20
    else:
        per_tile = 8 if "Li128ELi128E" in target["symbol"] else 24
    expected_mfma = target["shape"][2] // 128 * per_tile
    require(waves and len(waves) % expected_waves == 0, "Incomplete wave group count")
    totals = collections.defaultdict(lambda: [0, 0, 0])
    for o, w in zip(objs, waves):
        x = w["instructions"]
        require(o["num_insts"] == o["num_stitched"] == len(x)
                and o["duration"] == w["end"] - w["begin"], "Partial stitched wave")
        selected_cu = int(command["env"].get("ROCPROF_ATT_PARAM_TARGET_CU", "0"))
        require(w["cu"] == selected_cu and all(len(e) == 5 and e[3] >= e[2] >= 0 for e in x)
                and all(a[0] + a[2] <= b[0] for a, b in zip(x, x[1:])), "Wave issue order/CU differ")
        require(sum(code[e[4]][0].startswith("v_mfma_scale") for e in x) == expected_mfma, "Wave MFMA count differs from tile contract")
        for e in x:
            totals[e[4]][0] += 1
            totals[e[4]][1] += e[3]
            totals[e[4]][2] += e[2]
    for line, r in code.items():
        require(totals[line] == r[6:9], "Wave hit/duration/stall totals differ from code.json")
    kinds, deps, output_pc = make_kinds(code, waves)
    records, groups, all_tiles = [], [], []
    for path, w in zip(wave_paths, waves):
        x = w["instructions"]
        mfma = [e for e in x if kinds[e[4]] == "scaled_MFMA"]
        issues = [e[0] + e[2] for e in mfma]
        bars = [e for e in x if kinds[e[4]] == "barrier" and e[0] < issues[0]]
        require(bars, "No initial publication barrier")
        first_bar = bars[0]
        stores = [e for e in x if kinds[e[4]] == "C_VMEM_store"]
        conversion = next(e for e in x if kinds[e[4]] == "BF16_conversion")
        endpgm = next(e for e in x if kinds[e[4]] == "endpgm_decoder_duration")
        boundaries = [w["begin"], first_bar[0] + first_bar[3], issues[0], conversion[0], stores[0][0] + stores[0][2], endpgm[0], w["end"]]
        require(all(a <= b for a, b in zip(boundaries, boundaries[1:])), "Phase boundaries are out of order")
        labels = ["startup_through_initial_publication", "initial_LDS_and_tail_setup_to_first_MFMA",
                  "compute_to_first_output_conversion", "conversion_and_C_LDS_to_first_global_store",
                  "global_store_sequence_to_endpgm", "endpgm_attempt_to_wave_end"]
        phases = [{"phase": label, "clocks": b - a,
                   "share_of_wave_duration": (b - a) / (w["end"] - w["begin"]),
                   "breakdown": breakdown(x, kinds, a, b)}
                  for label, a, b in zip(labels, boundaries, boundaries[1:])]
        prologue_events = [event(e, code, kinds, w["begin"]) for e in x if e[0] < issues[0]
                           and (kinds[e[4]] != "other_instructions")]
        tiles = []
        tiles_count = len(issues) // per_tile
        for tile in range(len(issues) // per_tile):
            a = issues[tile * per_tile]
            b = issues[(tile + 1) * per_tile] if tile + 1 < tiles_count else conversion[0]
            # Final-tile BF16 output can start before its later MFMA events.
            mfma_window_end = issues[(tile + 1) * per_tile] if tile + 1 < tiles_count else issues[-1] + 4
            if tile == 0:
                scope = "startup_tile"
            elif tile == len(issues) // per_tile - 1:
                scope = "final_tile"
            elif tile >= len(issues) // per_tile - 2:
                scope = "penultimate_tile"
            else:
                scope = "interior_tiles"
            relevant = [e for e in x if a <= e[0] < mfma_window_end]
            scale_issues = [event(e, code, kinds, w["begin"]) for e in relevant
                            if kinds[e[4]] in ("SFA_scale_VMEM_issue", "SFB_scale_VMEM_issue")]
            item = {"tile": tile, "scope": scope, "first_MFMA_to_next_tile_or_conversion": b - a,
                    "first_to_last_tile_MFMA_issue": issues[(tile + 1) * per_tile - 1] - a,
                    "breakdown": breakdown(x, kinds, a, b), "scale_refill_issue_events": scale_issues,
                    "MFMA_window_clocks": mfma_window_end - a,
                    "MFMA_window_breakdown": breakdown(x, kinds, a, mfma_window_end)}
            tiles.append(item)
            all_tiles.append(item)
        wait_records = [{"code_line": line, "pc": hex(code[line][5]), "instruction": code[line][0],
                         "kind": kinds[line], "exported_queue_member_lines": sorted(members),
                         "exported_queue_member_instructions": [code[d][0] for d in sorted(members)]}
                        for line, members in sorted(deps.items())]
        record = {"wave": path.name, "wave_sha256": sha(path), "cu": w["cu"], "simd": w["simd"],
                  "slot": w["slot"], "id": w["id"], "begin": w["begin"], "end": w["end"],
                  "duration": w["end"] - w["begin"], "MFMA_count": len(mfma),
                  "first_MFMA": event(mfma[0], code, kinds, w["begin"]),
                  "first_MFMA_issue_from_wave_begin": issues[0] - w["begin"],
                  "first_to_last_MFMA_issue": issues[-1] - issues[0],
                  "last_MFMA_to_wave_end": w["end"] - issues[-1],
                  "last_MFMA_to_first_global_store": stores[0][0] + stores[0][2] - issues[-1],
                  "initial_barriers": [event(e, code, kinds, w["begin"]) for e in bars],
                  "initial_barrier_release_absolute": first_bar[0] + first_bar[3],
                  "phase_partition": phases, "prologue_memory_wait_events": prologue_events,
                  "whole_wave_breakdown": breakdown(x, kinds, w["begin"], w["end"]),
                  "decoder_duration_overlaps": [{"event": event(a, code, kinds, w["begin"]),
                                                   "next_event": event(b, code, kinds, w["begin"]),
                                                   "overlap_clocks": a[0] + a[3] - b[0]}
                                                  for a, b in zip(x, x[1:]) if a[0] + a[3] > b[0]],
                  "tile_records": tiles, "wait_dependency_queue_members": wait_records,
                  "timeline_sum_minus_wave_duration": sum(t[1] for t in w["timeline"]) - (w["end"] - w["begin"])}
        records.append(record)
    # Group by common initial barrier release; SIMD/slot alone does not identify a WG.
    for r in sorted(records, key=lambda r: r["initial_barrier_release_absolute"]):
        release = r["initial_barrier_release_absolute"]
        if not groups or release - groups[-1]["release_min"] > 4:
            groups.append({"release_min": release, "release_max": release, "waves": []})
        groups[-1]["release_max"] = release
        groups[-1]["waves"].append(r["wave"])
    require(all(len(g["waves"]) == expected_waves for g in groups), "Publication release groups are not complete workgroups")
    tile_stats = {}
    for scope in sorted({t["scope"] for t in all_tiles}):
        selected = [t for t in all_tiles if t["scope"] == scope]
        tile_stats[scope] = {"clocks": stats([t["first_MFMA_to_next_tile_or_conversion"] for t in selected]),
                             "breakdown_sum": dict(sum((collections.Counter(t["breakdown"]) for t in selected), collections.Counter())),
                             "scale_issue_events": sum(len(t["scale_refill_issue_events"]) for t in selected)}
    key_stats = {key: stats([r[key] for r in records]) for key in
                 ("duration", "first_MFMA_issue_from_wave_begin", "first_to_last_MFMA_issue", "last_MFMA_to_wave_end")}
    phase_stats = {label: stats([next(p["clocks"] for p in r["phase_partition"] if p["phase"] == label) for r in records])
                   for label in labels}
    require("torch" not in sys.modules, "CPU analysis imported torch")
    return {"target_index": index, "parent_id": target["kid"], "shape": target["shape"], "status": "passed",
            "purpose": target["purpose"], "symbol": target["symbol"], "traits": target["traits"],
            "identity": {"official_module_sha256": plan["official_binary_sha256"],
                         "application_sha256": sha(folder / "application.json"), "code_json_sha256": sha(ui / "code.json"),
                         "code_object": str(image), "code_object_sha256": sha(image), "FUNC_sha256": ref["instruction_sha256"],
                         "FUNC_bytes": size, "FUNC_vaddr": hex(begin), "metadata": metadata,
                         "descriptor_normalized_sha256": ref["descriptor_normalized_sha256"],
                         "all_code_instructions_match_capture_FUNC": True, "stats_CSV_matches_code_json": True,
                         "wave_hit_duration_stall_totals_match_code_json": True, "complete_MFMA_per_wave_checked": True},
            "clean_claim": claim, "wave_count": len(records), "waves_per_workgroup": expected_waves,
            "capture_name": name, "rejected_attempts": rejected_attempts or [],
            "MFMA_per_tile_per_wave": per_tile, "MFMA_expected_per_wave": expected_mfma,
            "inferred_workgroups_from_publication_release": groups,
            "wave_statistics": key_stats, "phase_clock_statistics": phase_stats,
            "tile_clock_statistics": tile_stats, "waves": records,
            "limits": ["Event participation does not prove nonzero EXEC or active-lane bytes; this decoder does not export EXEC masks.",
                       "Original durations are preserved for code/stat/wave aggregate checks. Phase partitions cap each event at the next event time and explicitly retain duration overlaps.",
                       "Workgroups inferred from synchronized publication release, not a fabricated SIMD-to-wave mapping.",
                       "One CU/SE and correlated waves do not establish whole-kernel timing gains.",
                       "Output phase begins at first BF16 conversion and overlaps final MFMAs for interleaved epilogues."]}


def main():
    plan, queue = read(HERE / "plan.json"), read(HERE / "att_queue.json")
    review = read(HERE.parent / "merged_review.json")
    audit = read(FORMAL / "identity_audit.json")
    require(sha(plan["official_binary"]) == plan["official_binary_sha256"] == review["baseline_official_module"]["sha256"], "Current official module changed")
    reference = {v["symbol"]: v for p in review["parents"] for v in p["variants"]}
    linked_shas = {c["candidate_image_sha256"] for c in audit["linked_module_bundle_checks"]}
    claim_path = HERE / "att_claim_corrected.jsonl"
    claims = [json.loads(s) for s in claim_path.read_text().splitlines()]
    recollect_queue = read(HERE / "recollect_queue.json") if (HERE / "recollect_queue.json").exists() else None
    recollect_claim_path = HERE / "recollect_claim.jsonl"
    recollect_claims = [json.loads(s) for s in recollect_claim_path.read_text().splitlines()] if recollect_claim_path.exists() else []
    captures, pending, errors = [], [], []
    for index, target in enumerate(plan["targets"]):
        if target["kid"] not in PARENTS:
            continue
        name = f"target{index}_kid{target['kid']}_att"
        folder = HERE / name
        current_queue, current_claims, suffix, rejected = queue, claims, "", []
        recollect_folder = HERE / (name + "_recollect")
        if target["kid"] == 9030 and list(recollect_folder.glob("ui_output*/code.json")):
            original_app = read(folder / "application.json")
            try:
                original_command = next(c for c in queue["commands"] if c["name"] == name)
                clean_claim(name, original_app, original_command, queue, claims)
            except ValueError as error:
                rejected.append({"capture_name": name, "reason": str(error), "application_sha256": sha(folder / "application.json")})
            folder, suffix = recollect_folder, "_recollect"
            current_queue, current_claims = recollect_queue, recollect_claims
        if not (folder / "application.json").exists() or not list(folder.glob("ui_output*/code.json")):
            pending.append({"target_index": index, "parent_id": target["kid"], "reason": "awaiting current code/wave capture"})
            continue
        app = read(folder / "application.json")
        if not any(c.get("event") == "end" and c.get("name") == name + suffix and c["time"] >= app.get("finished", float("inf")) for c in current_claims):
            pending.append({"target_index": index, "parent_id": target["kid"], "reason": "awaiting completed corrected claim"})
            continue
        try:
            captures.append(capture(index, target, plan, current_queue, current_claims, reference, linked_shas, suffix, rejected))
        except (ValueError, KeyError, StopIteration, FileNotFoundError) as error:
            errors.append({"target_index": index, "parent_id": target["kid"], "error": str(error)})
    result = {"status": "failed" if errors else ("partial_pending_captures" if pending else "passed"),
              "generated_utc": datetime.now(timezone.utc).isoformat(), "cpu_only": True,
              "new_GPU_execution": False, "new_build_execution": False, "production_sources_modified": False,
              "timing_contract": CONTRACT, "plan_sha256": sha(HERE / "plan.json"),
              "queue_sha256": sha(HERE / "att_queue.json"), "claim_path": str(claim_path),
              "claim_sha256_at_analysis": sha(claim_path), "analysis_script_sha256": sha(__file__),
              "recollect_claim_sha256_at_analysis": sha(recollect_claim_path) if recollect_claim_path.exists() else None,
              "review_sha256": sha(HERE.parent / "merged_review.json"),
              "official_module": review["baseline_official_module"], "captures": captures,
              "pending": pending, "errors": errors,
              "limits": ["No-profiler Event and physical-flow counters remain necessary before selecting an optimization.",
                         "Do not convert GRBM sums or ATT wave shares into absolute utilization/occupancy without a verified window.",
                         "9020 short/long use different bodies; 9022 short/long use one body; 9030 target is a K1536 actual winner."]}
    (HERE / "merged_att_analysis.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    lines = ["# Current Oct8 9020 / 9022 / 9030 ATT", "", CONTRACT, "",
             "Each accepted capture passed module/CO/FUNC/metadata/descriptor, current PC mapping, clean owner claim, complete wave MFMA, and code/stat/wave aggregate checks.", "",
             "| Target | Parent / shape | Waves / WGs | First MFMA median | Wave duration median | Last MFMA to end median |", "|---|---|---:|---:|---:|---:|"]
    for c in captures:
        s = c["wave_statistics"]
        lines.append(f"| {c['target_index']} | {c['parent_id']} / {'×'.join(map(str,c['shape']))} | {c['wave_count']} / {len(c['inferred_workgroups_from_publication_release'])} | {s['first_MFMA_issue_from_wave_begin']['median']} | {s['duration']['median']} | {s['last_MFMA_to_wave_end']['median']} |")
    lines += ["", "All numbers are shader clocks from correlated captured waves. Output begins at the first BF16 conversion and overlaps final MFMAs where the epilogue interleaves output. Full event, dependency, phase, and tile records are in [merged_att_analysis.json](merged_att_analysis.json).", "",
              f"Status: {result['status']}; pending {len(pending)}; errors {len(errors)}."]
    if errors:
        lines += ["", "Errors: " + json.dumps(errors, ensure_ascii=False)]
    (HERE / "merged_att_analysis.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": result["status"], "accepted_targets": [c["target_index"] for c in captures], "pending": pending, "errors": errors}, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
