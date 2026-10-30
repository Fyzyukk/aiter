#!/usr/bin/env python3
"""Preserve clean complete current-final shapes and resume on PCI-verified GPU5/6."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone

LOCAL = Path(__file__).resolve().parent
ROOT = LOCAL.parents[2]
SOURCE = LOCAL / "step4_resume_gpu47_20261030"
DEFAULT = LOCAL / "step4_final_recovery_gpu56_20261030"
FIELDS = ("gfx", "cu_num", "M", "N", "K", "w_scale_block")
PHYSICAL = [7, 4, 6, 5]
PCI = ["0000:f5:00.0", "0000:85:00.0", "0000:e5:00.0", "0000:95:00.0"]


def now():
    return datetime.now(timezone.utc).isoformat()


def key(row):
    return tuple(str(row[field]) for field in FIELDS)


def read(path):
    return json.loads(path.read_text())


def raw(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def dump(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def phase(output, value):
    (output / "phase").write_text(value + "\n")
    state = read(LOCAL / "workflow_state.json")
    state.update(updated_utc=now(), current_step=4, phase=value, gpu_ids=[5, 6])
    state["active_final_recovery"] = {"directory": str(output), "phase": value,
        "active_physical_gpu_ids": [5, 6], "logical_gpu_ids": [3, 2],
        "logical_to_physical": PHYSICAL, "logical_pci_buses": PCI}
    dump(LOCAL / "workflow_state.json", state)
    print(now(), value, flush=True)


def preflight(output):
    assert os.environ.get("ROCR_VISIBLE_DEVICES") == "4,5,6,7"
    assert os.environ.get("HIP_VISIBLE_DEVICES") == "0,1,2,3"
    snapshot = json.loads(subprocess.check_output(
        ["rocm-smi", "--showuse", "--showmeminfo", "vram", "--json"], text=True))
    buses = json.loads(subprocess.check_output(["rocm-smi", "--showbus", "--json"], text=True))
    for gpu in (5, 6):
        card = snapshot[f"card{gpu}"]
        assert int(card["GPU use (%)"]) <= 10, (gpu, card)
        assert int(card["VRAM Total Memory (B)"]) - int(card["VRAM Total Used Memory (B)"]) > 128 * 1024**3, (gpu, card)
    probe = '''import ctypes,json
h=ctypes.CDLL('/opt/rocm/lib/libamdhip64.so'); assert h.hipInit(0)==0
f=h.hipDeviceGetPCIBusId; f.argtypes=[ctypes.c_char_p,ctypes.c_int,ctypes.c_int]; f.restype=ctypes.c_int
rows=[]
for i in range(4):
 b=ctypes.create_string_buffer(32); assert f(b,32,i)==0; rows.append(b.value.decode().lower())
print(json.dumps(rows))
'''
    actual = json.loads(subprocess.check_output([sys.executable, "-c", probe], text=True))
    assert actual == PCI, actual
    assert all(buses[f"card{PHYSICAL[i]}"]["PCI Bus"].lower() == actual[i] for i in range(4))
    dump(output / "pci_preflight.json", {"checked_utc": now(), "logical_to_physical": PHYSICAL,
        "logical_pci_buses": actual, "active_physical_gpu_ids": [5, 6], "smi": snapshot, "buses": buses})


def prepare(output):
    from prepare_opus_repeat import registry_metadata, pure_common
    output.mkdir()
    for directory in ("repeats", "segments", "worker_plans"):
        (output / directory).mkdir()
    compile_receipt = read(LOCAL / "full421_union/compile_receipt.json")
    assert all(sha(Path(path)) == digest for path, digest in compile_receipt["source_sha256"].items())
    assert read(SOURCE / "final/repeats/status.json")["return_codes"] == {"0": -15, "1": 1, "2": 0, "3": 0}
    preflight(output)
    source_plan = read(SOURCE / "final_repeat_plan/repeat_plan.json")
    plan = json.loads(json.dumps(source_plan))
    items = {key(item["shape"]): item for item in plan["shapes"]}
    with (LOCAL.parent / "handoff_data/csv_baseline.csv").open() as stream:
        base_rows = list(csv.DictReader(stream))
    baseline = {key(row): float(row["us"]) for row in base_rows}
    assert len(baseline) == len(base_rows)
    metadata, pairs = registry_metadata()
    union = set(read(LOCAL / "full421_union/union.json")["union_kids"])
    outliers = [event for event in raw(SOURCE / "full421_top8/progress.jsonl")
        if (event["best_opus"]["us"] / baseline[key(event["shape"])] - 1) * 100 >= 20]
    assert len(outliers) == 104
    added = []
    for event in outliers:
        shape, best = key(event["shape"]), event["best_opus"]
        kid = best["kernelId"]
        if kid in {candidate["kid"] for candidate in items[shape]["candidates"]}:
            continue
        assert kid in union and metadata[kid]["kernelName"] == best["kernelName"]
        items[shape]["candidates"].append({**metadata[kid], "us": best["us"],
            "errRatio": best["errRatio"], "correct": True, "status": "correct",
            "roles": ["timing_review_initial_best"], "exact_opposite_kids": pairs[kid],
            "verified_evidence": [], "initial_source": {"path": str(SOURCE / "full421_top8/progress.jsonl")}})
        added.append({"shape": event["shape"], "kid": kid})
    assert len(added) == 6
    changed = {key(item["shape"]) for item in added}
    retained, source_hashes, preserved_records = {}, {}, 0
    for gpu in (2, 3):
        original = SOURCE / f"final/repeats/gpu{gpu}.jsonl"
        source_hashes[str(original)] = sha(original)
        rows = raw(original)
        groups = {}
        for row in rows:
            group = groups.setdefault((key(row["shape"]), row["kid"]), {})
            assert row["round"] not in group and row["correct"] and row["rotate_args"] == 0
            group[row["round"]] = row
        assert all(set(rounds) == set(range(6)) for rounds in groups.values())
        shapes = {shape for shape, _ in groups}
        assert len(shapes) == 105
        with (output / f"repeats/gpu{gpu}.jsonl").open("x") as stream:
            for line_number, row in enumerate(rows, 1):
                shape = key(row["shape"])
                if shape in changed:
                    continue
                retained[shape] = gpu
                row["original_declared_gpu_physical_id"] = row["gpu_physical_id"]
                row["gpu_physical_id"] = PHYSICAL[gpu]
                row["gpu_pci_bus"] = PCI[gpu]
                row["current_final_source_raw"] = {"path": str(original), "line": line_number,
                    "physical_mapping_evidence": str(output / "pci_preflight.json")}
                stream.write(json.dumps(row) + "\n")
                preserved_records += 1
    removed_clean_shapes = 210 - len(retained)
    assert 0 <= removed_clean_shapes <= len(changed)
    continuation = [item for shape, item in items.items() if shape not in retained]
    assert len(continuation) == 421 - len(retained)
    shards, loads = {2: [], 3: []}, {2: 0., 3: 0.}
    def cost(item):
        s = item["shape"]
        return len(item["candidates"]) * max(1., s["M"] * s["N"] * s["K"] / 1e10)
    for item in sorted(continuation, key=cost, reverse=True):
        gpu = min(shards, key=lambda index: (loads[index], len(shards[index])))
        shards[gpu].append(item)
        loads[gpu] += cost(item)
    plan["recovery"] = {"source_plan": str(SOURCE / "final_repeat_plan/repeat_plan.json"),
        "active_physical_gpus": [5, 6], "logical_to_physical": PHYSICAL, "added_timing_review_candidates": added}
    dump(output / "repeat_plan.json", plan)
    for gpu in shards:
        dump(output / f"worker_plans/gpu{gpu}_plan.json", {**plan, "shapes": shards[gpu],
            "gpu_logical_id": gpu, "gpu_physical_id": PHYSICAL[gpu], "gpu_pci_bus": PCI[gpu]})
    dump(output / "outliers104.json", outliers)
    receipt = {"created_utc": now(), "preserved_complete_shapes": len(retained), "repeating_shapes": len(continuation),
        "original_clean_complete_shapes": 210, "clean_shapes_repeated_with_added_candidate": removed_clean_shapes,
        "preserved_records": preserved_records, "full_plan_candidates": sum(len(i["candidates"]) for i in items.values()),
        "continuation_candidates": sum(len(i["candidates"]) for i in continuation),
        "source_hashes": source_hashes, "source_core_hashes": compile_receipt["source_sha256"],
        "physical_mapping": PHYSICAL, "active_physical_gpu_ids": [5, 6], "outlier_review_shapes": 104,
        "added_candidates": added,
        "provenance": "Complete current-final shapes from successful shards2/3 retained; raw physical metadata corrected using PCI mapping. All original raw files preserved. Each other shape reruns all candidates for six rounds in one session on GPU5/6. No historical step1/2/3 rounds supply final statistics."}
    dump(output / "recovery_receipt.json", receipt)
    for gpu in shards:
        subprocess.run([sys.executable, str(LOCAL / "repeat_opus_candidates.py"), "--plan",
            str(output / f"worker_plans/gpu{gpu}_plan.json"), "--output", str(output / f"segments/gpu{gpu}.jsonl"),
            "--gpu", str(gpu), "--rounds", "6", "--dry-run"], check=True)
    subprocess.run([sys.executable, str(LOCAL / "audit_final_repeat.py"), "--plan", str(output / "repeat_plan.json"),
        "--files", str(output / "repeats/gpu*.jsonl"), "--physical-gpus", "7,4,6,5",
        "--output-dir", str(output / "preserved_batch_audit")], check=True)
    phase(output, "prepared_final_recovery_gpu56")
    print(json.dumps(receipt, ensure_ascii=False, indent=2), flush=True)


def review(output):
    candidates = {}
    for path in (output / "repeats").glob("gpu*.jsonl"):
        for row in raw(path):
            assert row["gpu_physical_id"] in (5, 6) and row["gpu_pci_bus"] == PCI[row["gpu_logical_id"]]
            candidates.setdefault((key(row["shape"]), row["kid"]), []).append(row)
    results = []
    for event in read(output / "outliers104.json"):
        s, best = key(event["shape"]), event["best_opus"]
        selected = candidates[(s, best["kernelId"])]
        assert len(selected) == 6 and all(row["correct"] for row in selected)
        median = statistics.median(row["us"] for row in selected)
        baseline = selected[0]["csv_baseline_us"]
        winner = min(statistics.median(row["us"] for row in rows)
            for (shape, _), rows in candidates.items() if shape == s and len(rows) == 6 and all(row["correct"] for row in rows))
        results.append({"shape": event["shape"], "initial_best_kid": best["kernelId"], "initial_us": best["us"],
            "initial_slower_pct": (best["us"] / baseline - 1) * 100, "new_six_round_median_us": median,
            "new_six_round_min_us": min(row["us"] for row in selected), "new_six_round_max_us": max(row["us"] for row in selected),
            "new_six_round_slower_pct": (median / baseline - 1) * 100, "final_shape_best_us": winner,
            "final_shape_best_slower_pct": (winner / baseline - 1) * 100, "status": "six_round_checked"})
    dump(output / "outlier_review.json", {"checked_shapes": len(results), "results": results,
        "final_best_still_at_least20pct_slower": sum(r["final_shape_best_slower_pct"] >= 20 for r in results)})
    queue_path = LOCAL / "step4_screen208_corrected_losses80_20261030/pending_outlier_checks.json"
    queue = read(queue_path)
    reviewed = {key(item["shape"]): item for item in results}
    assert len(reviewed) == 104
    for item in queue["items"]:
        item["status"] = "final_six_round_checked"
        item["final_check"] = reviewed[key(item["shape"])]
    queue.update(status="six_round_checked", pending_shape_count=0, checked_shape_count=104,
        final_review_report=str(output / "outlier_review.json"), updated_utc=now())
    dump(queue_path, queue)
    note = ["# 第四步偏慢项六轮核对结果", "", "104项均已完成新最终六轮核对。",
        "仍明显偏慢项的原因需结合具体shape评估，不能将初筛偏慢直接归因于kernel。", "",
        "| M | N | K | 初筛多耗时 | 原候选六轮多耗时 | 最终最优多耗时 | 状态 |",
        "|---:|---:|---:|---:|---:|---:|---|"]
    for item in results:
        shape = item["shape"]
        note.append(f"| {shape['M']} | {shape['N']} | {shape['K']} | {item['initial_slower_pct']:.2f}% | "
            f"{item['new_six_round_slower_pct']:.2f}% | {item['final_shape_best_slower_pct']:.2f}% | 已核对 |")
    note += ["", "完整记录：" + str(output / "outlier_review.json"), ""]
    queue_path.with_name("pending_outlier_checks_zh.md").write_text("\n".join(note))


def run(output):
    with (output / "launcher.pid").open("x") as stream:
        stream.write(str(os.getpid()) + "\n")
    preflight(output)
    receipt = read(output / "recovery_receipt.json")
    assert all(sha(Path(path)) == digest for path, digest in receipt["source_core_hashes"].items())
    phase(output, "final_recovery_gpu56_six_rounds")
    processes, streams = {}, []
    try:
        for gpu in (2, 3):
            log = (output / f"segments/gpu{gpu}.log").open("x")
            streams.append(log)
            command = [sys.executable, str(LOCAL / "repeat_opus_candidates.py"), "--plan",
                str(output / f"worker_plans/gpu{gpu}_plan.json"), "--output", str(output / f"segments/gpu{gpu}.jsonl"),
                "--gpu", str(gpu), "--rounds", "6"]
            processes[gpu] = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            (output / f"segments/gpu{gpu}.pid").write_text(str(processes[gpu].pid) + "\n")
        statuses = {gpu: process.wait() for gpu, process in processes.items()}
    except BaseException:
        for process in processes.values():
            if process.poll() is None:
                process.terminate()
        raise
    finally:
        for log in streams:
            log.close()
    dump(output / "worker_status.json", statuses)
    if any(statuses.values()):
        phase(output, "final_recovery_worker_failed")
        raise SystemExit(1)
    for gpu in (2, 3):
        source = output / f"segments/gpu{gpu}.jsonl"
        with (output / f"repeats/gpu{gpu}.jsonl").open("a") as stream:
            for line_number, row in enumerate(raw(source), 1):
                assert row["gpu_pci_bus"] == PCI[gpu] and row["gpu_physical_id"] == PHYSICAL[gpu]
                row["current_final_source_raw"] = {"path": str(source), "line": line_number}
                stream.write(json.dumps(row) + "\n")
    phase(output, "final_recovery_batch_audit")
    subprocess.run([sys.executable, str(LOCAL / "audit_final_repeat.py"), "--plan", str(output / "repeat_plan.json"),
        "--files", str(output / "repeats/gpu*.jsonl"), "--physical-gpus", "7,4,6,5", "--require-complete",
        "--output-dir", str(output / "final_batch_audit")], check=True)
    review(output)
    phase(output, "final_recovery_report")
    subprocess.run([sys.executable, str(LOCAL / "finish_workflow.py"), "--profile",
        str(LOCAL / "focused_expansion_known_complete/profile.csv"), str(SOURCE / "full421_top8/profile.csv"),
        str(SOURCE / "winner_fallback_top0/profile.csv"), "--full-progress", str(SOURCE / "full421_top8/progress.jsonl"),
        "--fallback-progress", str(SOURCE / "winner_fallback_top0/progress.jsonl"), "--fallback-expected",
        str(LOCAL / "top8_public_after22_audit/winner_fallback_shapes.csv"), "--repeat", str(output / "repeats/gpu*.jsonl"),
        "--physical-gpus", "5,6", "--current-registry", "--require-repeat-winners", "--finalize",
        "--workflow-state", str(LOCAL / "workflow_state.json"), "--output-dir", str(output / "full421_final_report")], check=True)
    (output / "phase").write_text("complete\n")
    state = read(LOCAL / "workflow_state.json")
    state["active_final_recovery"]["phase"] = "complete"
    state["outlier_review_report"] = str(output / "outlier_review.json")
    state["pending_timing_outlier_checks"].update(pending_shapes=0, status="final_six_round_checked", checked_shapes=104)
    dump(LOCAL / "workflow_state.json", state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    parser.add_argument("--run", action="store_true")
    arguments = parser.parse_args()
    if arguments.run:
        try:
            run(arguments.output.resolve())
        except BaseException:
            (arguments.output / "exit_code").write_text("1\n")
            raise
        (arguments.output / "exit_code").write_text("0\n")
    else:
        prepare(arguments.output.resolve())
