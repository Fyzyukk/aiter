#!/usr/bin/env python3
"""Run at most four workers on owner-free physical AMD-SMI devices 0 through 3."""

import argparse
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
PREVIOUS = ROOT / "reports/opus_flydsl_comparison_20261008"
ALLOWED_SMI_INDICES = frozenset({0, 1, 2, 3})


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def select_devices(devices):
    """Fail closed for unknown activity, memory, HIP identity or process owner."""
    eligible = [
        gpu for gpu in devices
        if gpu["smi_index"] in ALLOWED_SMI_INDICES
        and gpu.get("hip_index") is not None
        and gpu.get("bdf")
        and gpu.get("peak_gfx") is not None
        and gpu["peak_gfx"] <= 2
        and gpu.get("used_gib") is not None
        and gpu["used_gib"] <= 8
        and gpu.get("free_gib") is not None
        and gpu["free_gib"] >= 16
        and gpu.get("processes") == []
    ]
    eligible.sort(key=lambda gpu: gpu["smi_index"])
    if len({gpu["hip_index"] for gpu in eligible}) != len(eligible):
        raise RuntimeError("Ambiguous SMI to HIP device enumeration")
    if len({gpu["bdf"] for gpu in eligible}) != len(eligible):
        raise RuntimeError("Ambiguous SMI to BDF device enumeration")
    return eligible


def fresh_path(path):
    path = Path(path).resolve()
    if not path.is_relative_to(OUT):
        raise ValueError(f"Output must stay in this continuation directory: {path}")
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    if not path.parent.is_dir():
        raise FileNotFoundError(f"Output directory is missing: {path.parent}")
    return path


def stop_child(child):
    if child.poll() is None:
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait()


def opus_identity(item):
    argv = item["argv"]
    plan_path = Path(argv[argv.index("--plan") + 1])
    plan = json.loads(plan_path.read_text())
    library = Path(plan["opus_library"])
    actual = hashlib.sha256(library.read_bytes()).hexdigest()
    if actual != plan["opus_sha256"]:
        raise RuntimeError(f"Selected OPUS library changed: {library}")
    return {"plan": str(plan_path.resolve()), "opus_library": str(library.resolve()),
            "opus_sha256": actual}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--wait-seconds", type=float, default=0,
                        help="Maximum idle wait; default takes one snapshot and exits 75 if blocked")
    parser.add_argument("--check-only", action="store_true",
                        help="Take one bounded snapshot; never acquire locks or start workers")
    args = parser.parse_args()
    if not math.isfinite(args.wait_seconds) or args.wait_seconds < 0:
        parser.error("--wait-seconds must be finite and non-negative")
    queue = json.loads(args.queue.read_text())
    commands = queue["commands"]
    if not 1 <= len(commands) <= 4:
        parser.error("Queue must contain between one and four workers")
    if len({item["name"] for item in commands}) != len(commands):
        parser.error("Worker names must be unique")
    claim_path = fresh_path(args.log)
    destinations = [claim_path]
    for item in commands:
        destinations.append(fresh_path(item["log"]))
        argv = item["argv"]
        if "--output" not in argv:
            parser.error("Every worker must declare its --output")
        result_path = fresh_path(argv[argv.index("--output") + 1])
        destinations.append(result_path)
        destinations.append(fresh_path(result_path.with_suffix(".receipt.json")))
    if len(set(destinations)) != len(destinations):
        parser.error("All claim, worker log and result paths must be distinct")

    owner = import_file("owner_queue_four", PREVIOUS / "run_when_idle_strict.py")
    picker = import_file("picker_four", ROOT / ".claude/skills/validate-kernel-pr/pick-idle-gpu.py")
    locks = []
    children = {}
    streams = {}
    receipts = set()
    failed = False
    smi = None
    initialized = False
    claim_stream = claim_path.open("x")

    def log(record):
        record["time"] = time.time()
        claim_stream.write(json.dumps(record) + "\n")
        claim_stream.flush()
        if record["event"] != "monitor":
            print(json.dumps(record), flush=True)

    def snapshot():
        devices, concurrent = picker.sample(smi, 3, 1)
        handles = smi.amdsmi_get_processor_handles()
        for gpu in devices:
            gpu["processes"] = smi.amdsmi_get_gpu_process_list(handles[gpu["smi_index"]])
        eligible = select_devices(devices)
        log({"event": "snapshot", "allowed_smi_indices": sorted(ALLOWED_SMI_INDICES),
             "eligible_smi_indices": [gpu["smi_index"] for gpu in eligible],
             "required_workers": len(commands), "other_peak_concurrent": concurrent,
             "devices": devices})
        return eligible, handles

    def recheck(gpu, handle):
        processes = smi.amdsmi_get_gpu_process_list(handle)
        gfx, umc = picker.read_activity(smi, handle)
        memory = smi.amdsmi_get_gpu_vram_usage(handle)
        current = {**gpu, "peak_gfx": gfx, "peak_umc": umc,
                   "used_gib": memory["vram_used"] / 1024,
                   "free_gib": (memory["vram_total"] - memory["vram_used"]) / 1024,
                   "processes": processes}
        if not select_devices([current]):
            log({"event": "occupied_before_launch", "gpu": current})
            return False
        return True

    def receipt(name, item, *, valid, status, returncode=None, external=None):
        output = Path(item["command"]["argv"][item["command"]["argv"].index("--output") + 1])
        record = {"name": name, "valid_for_analysis": valid, "status": status,
                  "returncode": returncode, "external_processes": external or [],
                  "result": str(output), "gpu": item["gpu"], "owner": item["owner"],
                  "fingerprint": item["fingerprint"], **item["opus_identity"], "time": time.time()}
        with output.with_suffix(".receipt.json").open("x") as stream:
            stream.write(json.dumps(record, indent=2) + "\n")
        receipts.add(name)
        log({"event": "analysis_receipt", **record})

    try:
        log({"event": "preflight", "queue": str(args.queue.resolve()),
             "queue_sha256": hashlib.sha256(args.queue.read_bytes()).hexdigest(),
             "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             "allowed_smi_indices": sorted(ALLOWED_SMI_INDICES),
             "wait_seconds": args.wait_seconds, "worker_count": len(commands),
             "check_only": args.check_only})
        smi = picker.import_amdsmi()
        smi.amdsmi_init()
        initialized = True
        deadline = time.monotonic() + args.wait_seconds
        while True:
            eligible, handles = snapshot()
            if args.check_only:
                code = 0 if len(eligible) >= len(commands) else 75
                log({"event": "preflight_only", "returncode": code, "workers_started": 0,
                     "eligible_smi_indices": [gpu["smi_index"] for gpu in eligible],
                     "required_workers": len(commands), "locks_acquired": 0})
                return code
            if len(eligible) >= len(commands):
                devices = eligible[:len(commands)]
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                log({"event": "blocked", "reason": "Allowed devices are not owner-free and idle",
                     "returncode": 75, "workers_started": 0})
                return 75
            time.sleep(min(15, remaining))

        for gpu in devices:
            fd = open(f"/tmp/gpu-{gpu['hip_index']}.lock", "a")
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                fd.close()
                log({"event": "blocked", "reason": "Physical GPU lock is held", "gpu": gpu,
                     "returncode": 75, "workers_started": 0})
                return 75
            locks.append(fd)
            if not recheck(gpu, handles[gpu["smi_index"]]):
                return 75
        log({"event": "claimed_four", "allowed_smi_indices": sorted(ALLOWED_SMI_INDICES),
             "devices": devices})
        environment = dict(os.environ)
        for key in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL"):
            environment.pop(key, None)
        environment.update(queue["env"])
        # Serialize KFD registration so each child has one uniquely identified host PID.
        for item, gpu in zip(commands, devices):
            if not recheck(gpu, handles[gpu["smi_index"]]):
                return 75
            child_environment = {**environment, "HIP_VISIBLE_DEVICES": str(gpu["hip_index"]),
                                 "OPUS_EXPECTED_GPU_BDF": gpu["bdf"]}
            identity_record = opus_identity(item)
            fingerprint = hashlib.sha256(json.dumps({
                "owner_command_fingerprint": owner.command_fingerprint(item, queue["env"]),
                **identity_record}, sort_keys=True).encode()).hexdigest()
            log({"event": "start", "command": item, "gpu": gpu,
                 "fingerprint": fingerprint, **identity_record})
            stream = Path(item["log"]).open("x")
            streams[item["name"]] = stream
            child, identity = owner.launch_owned_python(item, child_environment, stream)
            children[item["name"]] = {"child": child, "owner": identity, "gpu": gpu,
                                      "command": item, "fingerprint": fingerprint,
                                      "opus_identity": identity_record}
            log({"event": "owner_identity", "name": item["name"], **identity, "gpu": gpu})

        while children:
            for name, item in list(children.items()):
                child = item["child"]
                gpu = item["gpu"]
                processes = smi.amdsmi_get_gpu_process_list(handles[gpu["smi_index"]])
                foreign = [process for process in processes if process["pid"] != item["owner"]["host_pid"]]
                if foreign:
                    log({"event": "external_work_started", "name": name, "external": foreign, "gpu": gpu})
                    stop_child(child)
                    failed = True
                log({"event": "monitor", "name": name, "owner_host_pid": item["owner"]["host_pid"],
                     "gpu_bdf": gpu["bdf"], "smi_index": gpu["smi_index"],
                     "hip_index": gpu["hip_index"], "processes": processes})
                code = child.poll()
                if code is not None:
                    log({"event": "end", "name": name, "returncode": code,
                         "contamination": bool(foreign), "valid_for_analysis": code == 0 and not foreign,
                         "gpu": gpu})
                    receipt(name, item, valid=code == 0 and not foreign,
                            status="invalid_external_contamination" if foreign else
                                   ("completed" if code == 0 else "invalid_worker_failure"),
                            returncode=code, external=foreign)
                    streams[name].close()
                    del children[name]
                    if code:
                        failed = True
            if children:
                time.sleep(2)
        log({"event": "completed", "success": not failed})
        return int(failed)
    except BaseException as error:
        log({"event": "launcher_failed", "error": repr(error)})
        raise
    finally:
        for name, item in children.items():
            stop_child(item["child"])
            if name not in receipts:
                receipt(name, item, valid=False, status="invalid_launcher_interruption",
                        returncode=item["child"].poll())
        for stream in streams.values():
            stream.close()
        for fd in locks:
            fd.close()
        if initialized:
            smi.amdsmi_shut_down()
        claim_stream.close()


if __name__ == "__main__":
    raise SystemExit(main())
