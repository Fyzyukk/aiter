#!/usr/bin/env python3
"""Run a prepared command queue on a measured idle HIP GPU, with a device lock.

Uses AMD SMI only until a GPU is available. Stops its own child if another
process begins using that physical GPU; it never terminates external work.
"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import secrets
import select
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
OWNER_HELPERS = ROOT / "reports/opus_bound_analysis_20261007"
KFD_PROCESSES = Path("/sys/class/kfd/kfd/proc")


def registered_host_pids():
    return {int(path.name) for path in KFD_PROCESSES.iterdir() if path.name.isdecimal()}


def validate_owner(ready, *, child_pid, nonce, baseline, current, kfd_fd_target):
    if ready.get("inner_pid") != child_pid or ready.get("nonce") != nonce:
        raise RuntimeError("KFD identity handshake PID/nonce mismatch")
    if kfd_fd_target != "/dev/kfd":
        raise RuntimeError("Child did not hold the declared KFD fd")
    added = current - baseline
    if len(added) != 1:
        raise RuntimeError(f"KFD host PID identity is ambiguous: {sorted(added)}")
    return next(iter(added))


def launch_owned_python(item, environment, stream):
    """Identify a directly launched Python process before it imports HIP."""
    argv = item["argv"]
    if len(argv) < 2 or not argv[1].endswith(".py"):
        raise RuntimeError("Namespace-safe launcher supports direct Python scripts only")
    ready_read, ready_write = os.pipe()
    ack_read, ack_write = os.pipe()
    nonce = secrets.token_hex(24)
    bootstrap = OWNER_HELPERS / "owned_python_launch.py"
    baseline = registered_host_pids()
    child = None
    try:
        child_env = {**environment, "OPUS_OWNER_READY_FD": str(ready_write),
                     "OPUS_OWNER_ACK_FD": str(ack_read), "OPUS_OWNER_NONCE": nonce}
        child = subprocess.Popen([argv[0], str(bootstrap), *argv[1:]], cwd=ROOT,
                                 env=child_env, stdout=stream, stderr=subprocess.STDOUT,
                                 start_new_session=True, pass_fds=(ready_write, ack_read))
        os.close(ready_write)
        ready_write = None
        os.close(ack_read)
        ack_read = None
        if not select.select([ready_read], [], [], 10)[0]:
            raise RuntimeError("KFD identity handshake timed out")
        payload = os.read(ready_read, 8192)
        if not payload or child.poll() is not None:
            raise RuntimeError("Child exited before KFD identity handshake")
        ready = json.loads(payload)
        current = registered_host_pids()
        host_pid = validate_owner(ready, child_pid=child.pid, nonce=nonce,
                                 baseline=baseline, current=current,
                                 kfd_fd_target=os.readlink(f"/proc/{child.pid}/fd/{ready['kfd_fd']}"))
        if host_pid not in registered_host_pids() or child.poll() is not None:
            raise RuntimeError("KFD identity disappeared before acknowledgement")
        os.write(ack_write, (json.dumps({"nonce": nonce, "host_pid": host_pid}) + "\n").encode())
        return child, {"inner_pid": child.pid, "host_pid": host_pid,
                       "baseline_host_pids": sorted(baseline), "new_host_pids": sorted(current - baseline),
                       "launcher_sha256": hashlib.sha256(bootstrap.read_bytes()).hexdigest(),
                       "method": "KFD open + unique sysfs registration + same-mm runpy handshake"}
    except BaseException:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        raise
    finally:
        for fd in (ready_read, ready_write, ack_read, ack_write):
            if fd is not None:
                os.close(fd)

def command_fingerprint(item, environment):
    """CPU identity for resuming a successful command with unchanged inputs."""
    files = {}
    output_index = item["argv"].index("--output") + 1 if "--output" in item["argv"] else None
    pending = [Path(value) for index, value in enumerate(item["argv"])
               if index != output_index and Path(value).is_file()]
    pending.append(OWNER_HELPERS / "experiment_runner.py")
    if "--plan" in item["argv"]:
        plan = json.loads(Path(item["argv"][item["argv"].index("--plan") + 1]).read_text())
        pending.extend(Path(value) for value in plan.get("libraries", {}).values())
        if plan.get("official_binary"):
            pending.append(Path(plan["official_binary"]))
        for target in plan.get("targets", []):
            pending.extend(Path(value) for value in target.get("private_baselines", {}).values())
    for path in pending:
        if path.is_file():
            files[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    diff = subprocess.check_output(["git", "diff", "HEAD", "--", "csrc/opus_gemm"], cwd=ROOT)
    record = {"argv": item["argv"], "env": environment, "files": files,
              "source_head": source, "source_diff_sha256": hashlib.sha256(diff).hexdigest()}
    return hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()

def completed_fingerprints(path):
    current = {}
    completed = set()
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("event") == "start":
                current[record["command"]["name"]] = record.get("fingerprint")
            elif record.get("event") == "end" and record.get("returncode") == 0 and not record.get("contamination"):
                fingerprint = current.get(record["name"])
                if fingerprint:
                    completed.add(fingerprint)
    return completed

def interrupted_command_counts(path):
    counts = {}
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("event") == "end" and record.get("contamination"):
                name = record["name"]
                counts[name] = counts.get(name, 0) + 1
    return counts

def descendants(pid):
    found = {pid}
    pending = [pid]
    while pending:
        current = pending.pop()
        try:
            children = Path(f"/proc/{current}/task/{current}/children").read_text().split()
        except OSError:
            continue
        for child in children:
            value = int(child)
            if value not in found:
                found.add(value)
                pending.append(value)
    return found

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--idle-samples", type=int, default=10)
    parser.add_argument("--resume-completed", action="store_true")
    parser.add_argument("--retry-busy", action="store_true",
                        help="release device and wait again after external work interrupts this queue")
    args = parser.parse_args()
    queue = json.loads(args.queue.read_text())
    fingerprints = {item["name"]: command_fingerprint(item, {**queue.get("env", {}), **item.get("env", {})})
                    for item in queue["commands"]}
    completed = completed_fingerprints(args.log) if args.resume_completed else set()
    queue["commands"] = [item for item in queue["commands"] if fingerprints[item["name"]] not in completed]
    if queue.get("independent_commands_defer_interrupted", False):
        interruptions = interrupted_command_counts(args.log)
        queue["commands"].sort(key=lambda item: interruptions.get(item["name"], 0))
    if not queue["commands"]:
        print(json.dumps({"event": "all_commands_already_completed"}), flush=True)
        return 0
    args.log.parent.mkdir(parents=True, exist_ok=True)
    def log(record):
        record["time"] = time.time()
        with args.log.open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        if record.get("event") != "monitor":
            print(json.dumps(record), flush=True)

    spec = importlib.util.spec_from_file_location("idle_picker", ROOT / ".claude/skills/validate-kernel-pr/pick-idle-gpu.py")
    picker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(picker)
    smi = picker.import_amdsmi()
    smi.amdsmi_init()
    lock = None
    child = None
    try:
        handle = None
        while handle is None:
            gpus, concurrent = picker.sample(smi, args.idle_samples, 1)
            candidates = [gpu for gpu in gpus if gpu["hip_index"] is not None
                          and gpu["peak_gfx"] is not None and gpu["peak_gfx"] <= 2
                          and gpu["used_gib"] <= 8 and gpu["free_gib"] >= 16]
            candidates.sort(key=lambda gpu: (gpu["used_gib"], gpu["mean_gfx"]))
            for gpu in candidates:
                candidate_lock = open(f"/tmp/gpu-{gpu['hip_index']}.lock", "w")
                try:
                    fcntl.flock(candidate_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    candidate_lock.close()
                    continue
                candidate_handle = smi.amdsmi_get_processor_handles()[gpu["smi_index"]]
                gfx, _ = picker.read_activity(smi, candidate_handle)
                usage = smi.amdsmi_get_gpu_vram_usage(candidate_handle)["vram_used"] / 1024
                if gfx is None or gfx > 2 or usage > 8:
                    candidate_lock.close()
                    continue
                lock = candidate_lock
                handle = candidate_handle
                selected = gpu
                break
            if handle is None:
                log({"event": "waiting", "peak_gfx": [gpu["peak_gfx"] for gpu in gpus],
                     "used_gib": [round(gpu["used_gib"], 2) for gpu in gpus]})
                time.sleep(20)
        log({"event": "claimed", "gpu": selected, "other_busy": concurrent})
        environment = os.environ.copy()
        for key in ("ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL"):
            environment.pop(key, None)
        environment.update(queue.get("env", {}))
        environment["HIP_VISIBLE_DEVICES"] = str(selected["hip_index"])
        environment["OPUS_EXPECTED_GPU_BDF"] = selected["bdf"]
        for item in queue["commands"]:
            gfx, _ = picker.read_activity(smi, handle)
            usage = smi.amdsmi_get_gpu_vram_usage(handle)["vram_used"] / 1024
            if gfx is None or gfx > 2 or usage > 8:
                log({"event": "occupied_before_next", "gfx": gfx, "used_gib": usage})
                return 75
            output = Path(item["log"])
            output.parent.mkdir(parents=True, exist_ok=True)
            artifact = None
            if "--output" in item["argv"]:
                artifact = Path(item["argv"][item["argv"].index("--output") + 1])
            # Preserve earlier attempts before a prepared queue is restarted.
            for existing in (output, artifact):
                if existing is not None and existing.exists():
                    stamp = str(time.time_ns())
                    existing.rename(existing.with_name(existing.stem + ".attempt_" + stamp + existing.suffix))
            log({"event": "start", "command": item, "fingerprint": fingerprints[item["name"]]})
            previous_gfx = {process["pid"]: process.get("engine_usage", {}).get("gfx")
                            for process in smi.amdsmi_get_gpu_process_list(handle)}
            with output.open("w") as stream:
                child_environment = {**environment, **item.get("env", {})}
                try:
                    child, owner = launch_owned_python(item, child_environment, stream)
                except (OSError, ValueError, RuntimeError, KeyError) as error:
                    log({"event": "owner_identity_unresolved", "name": item["name"], "error": str(error)})
                    log({"event": "end", "name": item["name"], "returncode": 75, "contamination": True})
                    return 75
                log({"event": "owner_identity", "name": item["name"], **owner})
                contamination = False
                while child.poll() is None:
                    processes = smi.amdsmi_get_gpu_process_list(handle)
                    external = []
                    for process in processes:
                        if process["pid"] == owner["host_pid"]:
                            continue
                        external.append(process)
                    previous_gfx = {process["pid"]: process.get("engine_usage", {}).get("gfx")
                                    for process in processes}
                    if external:
                        contamination = True
                        log({"event": "external_work_started", "external": external, "child_pid": child.pid})
                        os.killpg(child.pid, signal.SIGTERM)
                        try:
                            child.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid, signal.SIGKILL)
                            child.wait()
                        break
                    log({"event": "monitor", "child_pid": child.pid,
                         "processes": [{"pid": process["pid"], "vram": process.get("memory_usage", {}).get("vram_mem")}
                                       for process in processes]})
                    time.sleep(2)
                return_code = child.wait()
            child = None
            log({"event": "end", "name": item["name"], "returncode": return_code, "contamination": contamination})
            if contamination and artifact is not None and artifact.exists():
                record = json.loads(artifact.read_text())
                record["status"] = "interrupted_external_gpu_work"
                record["contamination"] = {"detected_at_unix": time.time(),
                    "external_pids": [process["pid"] for process in external],
                    "claim_log": str(args.log.resolve()), "timings_usable_for_final_decision": False}
                artifact.write_text(json.dumps(record, indent=2) + "\n")
            if return_code or contamination:
                return 75 if contamination else return_code
            # Allow the process's context and activity sampling to settle.
            time.sleep(3)
        log({"event": "completed"})
        return 0
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        if lock is not None:
            lock.close()
        smi.amdsmi_shut_down()

if __name__ == "__main__":
    while True:
        code = main()
        if code != 75 or "--retry-busy" not in os.sys.argv:
            raise SystemExit(code)
        if "--resume-completed" not in os.sys.argv:
            os.sys.argv.append("--resume-completed")
        print(json.dumps({"event": "retry_wait_after_external_work"}), flush=True)
        time.sleep(20)
