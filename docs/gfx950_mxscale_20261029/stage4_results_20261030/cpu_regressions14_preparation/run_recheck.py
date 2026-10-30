#!/usr/bin/env python3
"""Prepare, or explicitly execute, the frozen exact-Opus regression recheck.

The default, plan, and check paths use only the Python standard library.  GPU
imports and device discovery exist exclusively inside execute_gpu(), behind
the execute action, --execute-gpu, and an explicit physical --gpu selection.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import ctypes.util
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time


BUNDLE = Path(__file__).resolve().parent
SHAPE_FIELDS = ("gfx", "cu_num", "M", "N", "K", "w_scale_block")
PAIR_FIELDS = (*SHAPE_FIELDS, "libtype", "kernelId", "kernelName", "splitK", "role")
ITERS, WARMUP, BATCHES, SEED = 101, 5, 3, 0
ERR_LIMIT = 0.05
SCHEMA_VERSION = 1


class RecheckError(RuntimeError):
    pass


def timestamp():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_fingerprints(fingerprints):
    for path, expected in fingerprints.items():
        actual = sha256_file(path)
        if actual != expected:
            raise RecheckError(f"Frozen file changed: {path}; {actual} != {expected}")


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def shape_key(pair):
    return tuple(int(pair[field]) for field in ("M", "N", "K"))


def identity(pair, batch):
    return (*shape_key(pair), int(pair["kernelId"]), batch)


def shape_label(shape):
    return "x".join(str(value) for value in shape)


def scheduled_rows(pairs, shape_order):
    grouped = {tuple(shape): [] for shape in shape_order}
    for pair in pairs:
        grouped[shape_key(pair)].append(pair)
    result = []
    for shape in grouped:
        for batch in range(1, BATCHES + 1):
            ordered = grouped[shape] if batch != 2 else list(reversed(grouped[shape]))
            for order, pair in enumerate(ordered):
                result.append((pair, batch, order))
    return grouped, result


def canonical_config(rotate_args):
    return {
        "iters": ITERS,
        "warmup": WARMUP,
        "batches": BATCHES,
        "seed": SEED,
        "rtol": 0.01,
        "atol": 0.01,
        "errRatio_limit": ERR_LIMIT,
        "rotate_args": rotate_args,
        "rotation_policy": "explicit-fixed" if rotate_args else "shared-auto",
        "candidate_order": "AB/BA/AB",
        "single_gpu_serial": True,
        "rebuild_allowed": False,
    }


def validate_selection(preflight, scope):
    pairs = preflight["selected_pairs"]
    shape_order = [tuple(int(value) for value in shape) for shape in preflight["shape_order"]]
    expected_pairs, expected_shapes = {"all": (26, 14), "priority": (4, 2),
                                       "remaining": (22, 12)}[scope]
    if len(pairs) != expected_pairs or len(shape_order) != expected_shapes:
        raise RecheckError(f"Unexpected {scope} selection cardinality")
    selected = set()
    for pair in pairs:
        for field in PAIR_FIELDS:
            if field not in pair:
                raise RecheckError(f"Missing pair field: {field}")
        if (str(pair["gfx"]), int(pair["cu_num"]), str(pair["w_scale_block"]),
                str(pair["libtype"]), int(pair["splitK"])) != (
                    "gfx950", 256, "128x128", "opus", 1):
            raise RecheckError("Only frozen gfx950/256/128x128 exact Opus pairs are supported")
        if not pair["role"] or not pair["kernelName"] or int(pair["kernelId"]) < 0:
            raise RecheckError("Invalid exact candidate identity")
        key = identity(pair, 1)
        if key in selected:
            raise RecheckError(f"Duplicate shape/kernel pair: {key}")
        selected.add(key)
    if set(shape_order) != {shape_key(pair) for pair in pairs} or len(set(shape_order)) != len(shape_order):
        raise RecheckError("shape_order does not match selected pairs")
    return pairs, shape_order


def load_preflight(scope):
    # cpu_preflight is itself a stdlib-only module; do not load the tuner here.
    preflight_module = importlib.import_module("cpu_preflight")
    checked = preflight_module.validate_bundle(scope=scope)
    pairs, shape_order = validate_selection(checked, scope)
    verify_fingerprints(checked["fingerprints"])
    return preflight_module, checked, pairs, shape_order


def describe_plan(checked, pairs, shape_order, args):
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "cpu_preflight_ok",
        "scope": args.scope,
        "gpu_used": False,
        "selected_shapes": len(shape_order),
        "selected_pairs": len(pairs),
        "planned_results": len(pairs) * BATCHES,
        "config": canonical_config(args.rotate_args),
        "base_opus_calls": len(pairs) * BATCHES * (ITERS + WARMUP),
        "shared_auto_probe_calls": 0 if args.rotate_args else len(pairs),
        "reference_gemms": len(shape_order),
        "output_dir": str(args.output_dir or BUNDLE / ("recheck_" + args.scope)),
        "fingerprints": checked["fingerprints"],
        "preparation_plan": checked["plan"],
        "execution_requires": "execute --execute-gpu --gpu {4,5,6,7}",
    }


def row_base(pair, batch, order):
    row = {field: pair[field] for field in PAIR_FIELDS}
    for field in ("cu_num", "M", "N", "K", "kernelId", "splitK"):
        row[field] = int(row[field])
    row.update(batch=batch, order=order, seed=SEED)
    return row


def read_resume_rows(path, schedule, rotations):
    expected = {identity(pair, batch): (pair, order) for pair, batch, order in schedule}
    completed = {}
    if not path.exists():
        return completed
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                raise RecheckError(f"Blank results line {line_number}; repair explicitly before resuming")
            try:
                row = json.loads(line)
                integer_fields = ("cu_num", "M", "N", "K", "kernelId", "splitK",
                                  "batch", "order", "seed")
                if any(type(row.get(field)) is not int for field in integer_fields):
                    raise ValueError("identity/order/seed fields must be JSON integers, not bool or float")
                batch = row["batch"]
                if type(batch) is not int or batch not in (1, 2, 3):
                    raise ValueError("invalid batch")
                key = identity(row, batch)
                pair, order = expected[key]
                base = row_base(pair, batch, order)
                if any(row.get(field) != value for field, value in base.items()):
                    raise ValueError("row identity/order/seed/role mismatch")
                if key in completed:
                    raise ValueError("duplicate row")
                if row.get("status") == "error":
                    raise RecheckError(f"Existing error row {line_number} is a failure, not resumable success")
                if row.get("status") != "ok":
                    raise ValueError("invalid status")
                us, error = row["us"], row["errRatio"]
                if isinstance(us, bool) or not isinstance(us, (int, float)) or not math.isfinite(us) or us <= 0:
                    raise ValueError("invalid latency")
                if isinstance(error, bool) or not isinstance(error, (int, float)) or not math.isfinite(error) or not 0 <= error <= ERR_LIMIT:
                    raise ValueError("invalid correctness ratio")
                shape_rotation = rotations[shape_label(shape_key(pair))]
                if row.get("rotation_args") != shape_rotation["num_rotate_args"]:
                    raise ValueError("rotation differs from frozen manifest")
            except RecheckError:
                raise
            except (KeyError, TypeError, ValueError) as error:
                raise RecheckError(f"Invalid results line {line_number}: {error}") from error
            completed[key] = row
    return completed


def prepare_output(preflight_module, checked, pairs, shape_order, args):
    output_dir = Path(args.output_dir or BUNDLE / ("recheck_" + args.scope)).resolve()
    config = canonical_config(args.rotate_args)
    gpu = {
        "physical_id": args.gpu,
        "uuid": preflight_module.GPU_UUIDS[args.gpu],
        "expected_pci_bus": preflight_module.GPU_BUSES[args.gpu],
    }
    _, schedule = scheduled_rows(pairs, shape_order)
    manifest_path = output_dir / "manifest.json"
    if args.resume:
        if not manifest_path.is_file():
            raise RecheckError("--resume requires an existing manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") == "error":
            raise RecheckError("Existing manifest records a failed run; use a new output directory after investigating it")
        comparisons = {"schema_version": SCHEMA_VERSION, "scope": args.scope,
                       "config": config, "fingerprints": checked["fingerprints"],
                       "gpu": gpu}
        if any(manifest.get(field) != expected for field, expected in comparisons.items()):
            raise RecheckError("Resume scope/config/fingerprints/GPU do not match the frozen manifest")
        with (output_dir / "selected_pairs.csv").open(newline="", encoding="utf-8") as stream:
            persisted_pairs = list(csv.DictReader(stream))
        if persisted_pairs != pairs:
            raise RecheckError("Resume selected_pairs.csv differs from the frozen scope")
        completed = read_resume_rows(output_dir / "results.jsonl", schedule,
                                     manifest.get("shape_rotations", {}))
        manifest.setdefault("resumes", []).append({"utc": timestamp(), "valid_rows": len(completed)})
    else:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise RecheckError("Output directory is nonempty; use a new path or explicit --resume")
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "scope": args.scope,
            "config": config,
            "fingerprints": checked["fingerprints"],
            "gpu": gpu,
            "shape_order": [list(shape) for shape in shape_order],
            "planned_results": len(schedule),
            "shape_rotations": {},
            "created_utc": timestamp(),
            "cpu_gate": {"action": "execute", "execute_gpu": True,
                         "explicit_physical_gpu": args.gpu, "preflight": "passed"},
            "source_preparation_plan": checked["plan"],
        }
        with (output_dir / "selected_pairs.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(pairs[0]))
            writer.writeheader()
            writer.writerows(pairs)
            stream.flush()
            os.fsync(stream.fileno())
        completed = {}
    atomic_json(manifest_path, manifest)
    return output_dir, manifest, completed, schedule


def hip_pci_bus_id(logical_device):
    library_name = ctypes.util.find_library("amdhip64") or "/opt/rocm/lib/libamdhip64.so"
    library = ctypes.CDLL(library_name)
    function = library.hipDeviceGetPCIBusId
    function.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
    function.restype = ctypes.c_int
    buffer = ctypes.create_string_buffer(64)
    result = function(buffer, len(buffer), logical_device)
    if result != 0:
        raise RecheckError(f"hipDeviceGetPCIBusId failed with HIP status {result}")
    return buffer.value.decode("ascii")


def parse_gpu_occupancy(output):
    """Parse the same rocm-smi PID format as continue_stages, fail closed."""
    declarations = list(re.finditer(
        r"^PID (\d+) is using (\d+) DRM device\(s\):?[ \t]*$",
        output, re.MULTILINE))
    if any(marker in output for marker in ("Unable to get", "ERROR", "Error:", "error:")):
        raise RecheckError("rocm-smi reported an occupancy query error")
    if not declarations:
        if "No KFD PIDs currently running" in output:
            return {}
        raise RecheckError("Unexpected rocm-smi occupancy output")
    if len(re.findall(r"^PID\b", output, re.MULTILINE)) != len(declarations):
        raise RecheckError("Unparseable PID declaration in rocm-smi output")
    occupied = {}
    for declaration in declarations:
        pid, count = int(declaration[1]), int(declaration[2])
        if pid in occupied:
            raise RecheckError("Duplicate PID declaration in rocm-smi output")
        if count == 0:
            occupied[pid] = set()
            continue
        following = output[declaration.end():].lstrip("\r\n")
        indexes = following.splitlines()[0].strip() if following else ""
        if not re.fullmatch(r"\d+(?:[ \t]+\d+)*", indexes):
            raise RecheckError("Cannot interpret GPU process occupancy")
        devices = [int(value) for value in indexes.split()]
        if len(devices) != count or len(set(devices)) != count:
            raise RecheckError("GPU occupancy count does not match device indices")
        occupied[pid] = set(devices)
    return occupied


def gpu_occupancy_snapshot():
    # Only called inside explicit execution, never by CPU preflight/check/plan.
    query_environment = os.environ.copy()
    for name in ("ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL"):
        query_environment.pop(name, None)
    command = "/opt/rocm/bin/rocm-smi"
    result = subprocess.run([command, "--showpidgpus"], capture_output=True,
                            text=True, check=False, timeout=30, env=query_environment)
    if result.returncode:
        raise RecheckError(f"rocm-smi occupancy query failed: {result.stderr.strip()}")
    occupied = parse_gpu_occupancy(result.stdout)
    return {"checked_utc": timestamp(),
            "known_pid_devices": {str(pid): sorted(devices) for pid, devices in occupied.items()}}


def assert_gpu_idle(physical_gpu, ignore_pid=None):
    snapshot = gpu_occupancy_snapshot()
    occupied = {int(pid): set(devices) for pid, devices in snapshot["known_pid_devices"].items()}
    other_pids = [pid for pid, devices in occupied.items()
                  if pid != ignore_pid and physical_gpu in devices]
    if other_pids:
        raise RecheckError(f"Physical GPU {physical_gpu} is occupied by PIDs {other_pids}")
    return {**snapshot, "physical_gpu": physical_gpu,
            "selected_gpu_idle": True,
            "ignored_own_driver_pid": ignore_pid}


def resolve_driver_ownership(physical_gpu):
    """Resolve the driver's PID after registering our sole visible context.

    rocm-smi can report host PIDs while /proc exposes container PIDs.  Never
    identify a KFD process by comparing its number with os.getpid().  The card
    was required to be idle before import; one process after our forced
    allocation is our driver process.  Multiple processes refuse execution.
    """
    snapshot = gpu_occupancy_snapshot()
    pids = [int(pid) for pid, devices in snapshot["known_pid_devices"].items()
            if physical_gpu in devices]
    if len(pids) > 1:
        raise RecheckError(f"GPU {physical_gpu} has multiple KFD PIDs after context registration: {pids}")
    return {**snapshot, "physical_gpu": physical_gpu, "container_pid": os.getpid(),
            "driver_pid": pids[0] if pids else None,
            "status": "resolved" if pids else "unresolved",
            "method": "Initially idle card; tiny allocation/sync; exactly one selected-card KFD PID",
            "active_policy": "Exclude resolved driver PID" if pids else
                             "Snapshot only; driver PID unavailable; no namespace identity guessed"}


def check_active_gpu(physical_gpu, ownership):
    driver_pid = ownership["driver_pid"]
    if driver_pid is not None:
        return assert_gpu_idle(physical_gpu, ignore_pid=driver_pid)
    snapshot = gpu_occupancy_snapshot()
    return {**snapshot, "physical_gpu": physical_gpu,
            "driver_ownership": "unresolved", "active_check": "snapshot-only",
            "selected_gpu_idle": None,
            "note": "Only the pre-import idle gate was enforceable; no container/host PID identity guessed"}


def confirm_device(torch, expected_bus):
    if torch.cuda.device_count() != 1:
        raise RecheckError("Execution requires exactly one visible GPU")
    torch.cuda.set_device(0)
    properties = torch.cuda.get_device_properties(0)
    architecture = str(getattr(properties, "gcnArchName", "")).split(":", 1)[0]
    cu_num = int(getattr(properties, "multi_processor_count", 0))
    pci = hip_pci_bus_id(0)
    bus = int(pci.rsplit(":", 2)[-2], 16)
    if isinstance(expected_bus, str) and ":" in expected_bus:
        expected_number = int(expected_bus.rsplit(":", 2)[-2], 16)
    else:
        expected_number = int(expected_bus)
    if architecture != "gfx950" or cu_num != 256 or bus != expected_number:
        raise RecheckError(f"GPU identity mismatch: arch={architecture}, CUs={cu_num}, PCI={pci}")
    return {"logical_device": 0, "gfx": architecture, "cu_num": cu_num,
            "pci_bus_id": pci, "name": str(properties.name)}


def finite_number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


class TraceCapture:
    """Proxy the existing profiler without changing its event/filtering API."""

    def __init__(self, test_common, output_dir):
        self.module = test_common.tpf
        self.original = self.module.profile
        self.output_dir = output_dir
        self.active = None
        self.records = []

    def __enter__(self):
        self.module.profile = self.factory
        return self

    def __exit__(self, *exception):
        self.module.profile = self.original

    def factory(self, *args, **kwargs):
        owner = self
        inner = self.original(*args, **kwargs)

        class Proxy:
            def __enter__(self):
                inner.__enter__()
                return self

            def __exit__(self, *exception):
                result = inner.__exit__(*exception)
                owner.capture(inner)
                return result

            def __getattr__(self, name):
                return getattr(inner, name)

        return Proxy()

    def capture(self, profiler):
        if self.active is None:
            return
        record = {"trace_path": None, "raw_iteration_us": [], "capture_error": None}
        try:
            events = []
            for event in profiler.events():
                device_type = str(getattr(event, "device_type", ""))
                if not device_type.endswith(("CUDA", "HIP")):
                    continue
                events.append({
                    "name": str(event.name),
                    "device_type": device_type,
                    "device_index": int(getattr(event, "device_index", -1)),
                    "self_device_time_total_us": finite_number(getattr(event, "self_device_time_total", None)),
                    "device_time_total_us": finite_number(getattr(event, "device_time_total", None)),
                })
            samples, grouping_note = [], "Device event count/pattern cannot be grouped into 101 calls"
            if events and len(events) % ITERS == 0:
                kernels_per_call = len(events) // ITERS
                first_names = [event["name"] for event in events[:kernels_per_call]]
                groups = [events[index:index + kernels_per_call]
                          for index in range(0, len(events), kernels_per_call)]
                if all([event["name"] for event in group] == first_names and
                       all(event["self_device_time_total_us"] is not None for event in group)
                       for group in groups):
                    samples = [sum(event["self_device_time_total_us"] for event in group)
                               for group in groups]
                    grouping_note = "Consecutive equal kernel patterns; first call is retained in raw samples"
            metadata = self.active
            name = (f"batch{metadata['batch']}_M{metadata['M']}_N{metadata['N']}_"
                    f"K{metadata['K']}_kid{metadata['kernelId']}_profile{len(self.records)}.json")
            trace_path = self.output_dir / "traces" / name
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"schema_version": SCHEMA_VERSION, "identity": metadata,
                       "device_events": events, "raw_iteration_us": samples,
                       "grouping_note": grouping_note,
                       "measurement_note": "Original get_trace_perf removes first call and applies its IQR policy unchanged"}
            atomic_json(trace_path, payload)
            record["trace_path"] = str(trace_path.relative_to(self.output_dir))
            record["raw_iteration_us"] = samples
            record["grouping_note"] = grouping_note
        except Exception as error:
            # Raw trace capture is auxiliary; preserve the original benchmark.
            record["capture_error"] = f"{type(error).__name__}: {error}"
        self.records.append(record)


def probe_shared_rotation(torch, test_common, tuner_module, data, pairs):
    probes = []
    for pair in pairs:
        arguments = tuple(data[field] for field in ("x", "weight_shuffle", "x_scale", "w_scale", "out")) + (int(pair["kernelId"]),)
        try:
            used, input_size, peak, non_torch = test_common.device_memory_profiling(
                tuner_module.run_gemm_opus, *arguments)
            torch.cuda.synchronize()
            properties = torch.cuda.get_device_properties(0)
            free_memory = torch.cuda.mem_get_info(0)[0]
            l2_size = getattr(properties, "L2_cache_size", 4096 * 1024)
            cache_size = max(0, min(l2_size * 64 * 128,
                                    (free_memory - used + input_size) * 0.9))
            source_num = min(int((cache_size + input_size - 1) // input_size), ITERS)
            probes.append({"kernelId": int(pair["kernelId"]),
                           "input_bytes": int(input_size), "iter_used_bytes": int(used),
                           "peak_increase_bytes": int(peak), "non_torch_increase_bytes": int(non_torch),
                           "free_bytes": int(free_memory), "l2_bytes": int(l2_size),
                           "cache_target_bytes": int(cache_size),
                           "source_num": source_num, "effective_num": max(1, source_num)})
        except Exception as error:
            raise RecheckError(f"Rotation probe failed for kid {pair['kernelId']}: {error}") from error
    return {"policy": "shared-auto", "num_rotate_args": min(probe["effective_num"] for probe in probes),
            "probes": probes, "recorded_utc": timestamp()}


def append_row(stream, row):
    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def execute_gpu(preflight_module, checked, pairs, shape_order, args):
    if args.action != "execute" or not args.execute_gpu or args.gpu is None:
        raise RecheckError("GPU execution requires execute --execute-gpu --gpu {4,5,6,7}")
    output_dir, manifest, completed, schedule = prepare_output(
        preflight_module, checked, pairs, shape_order, args)
    manifest_path = output_dir / "manifest.json"
    verify_fingerprints(checked["fingerprints"])
    if len(completed) == len(schedule):
        print(json.dumps({"status": "already_complete", "gpu_used": False,
                          "valid_rows": len(completed), "output_dir": str(output_dir)}))
        return 0

    try:
        # Query physical-card occupancy before any Torch import/device context.
        manifest["initial_occupancy_check"] = assert_gpu_idle(args.gpu)
        atomic_json(manifest_path, manifest)
        os.environ["ROCR_VISIBLE_DEVICES"] = str(preflight_module.GPU_UUIDS[args.gpu])
        for name in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL"):
            os.environ.pop(name, None)
        os.environ["AITER_JIT_DIR"] = str(preflight_module.JIT_ROOT)
        os.environ["OPUS_GEN_CO_DIR"] = str(preflight_module.CO_ROOT.parent)
        os.environ["AITER_REBUILD"] = "0"
        os.environ["AITER_LOG_MORE"] = "0"
        sys.path.insert(0, str(preflight_module.REPO))
        torch = importlib.import_module("torch")
        jit_core = importlib.import_module("aiter.jit.core")

        def deny_build(*unused_args, **unused_kwargs):
            raise RecheckError("Compilation is forbidden in this frozen-SO recheck")

        jit_core.AITER_REBUILD = 0
        jit_core.build_module = deny_build
        tuner_module = importlib.import_module(
            "csrc.gemm_a8w8_mxfp8_bpreshuffle.gemm_a8w8_mxfp8_bpreshuffle_tune")
        worker = importlib.import_module("aiter.utility.mp_tuner").worker
        test_common = importlib.import_module("aiter.test_common")
        manifest["observed_device"] = confirm_device(torch, preflight_module.GPU_BUSES[args.gpu])
        registration_tensor = torch.empty(1, dtype=torch.uint8, device="cuda:0")
        torch.cuda.synchronize()
        manifest["driver_ownership"] = resolve_driver_ownership(args.gpu)
        registration_tensor = None
        manifest["status"] = "running"
        manifest["execution_started_utc"] = timestamp()
        atomic_json(manifest_path, manifest)

        grouped, _ = scheduled_rows(pairs, shape_order)
        with (output_dir / "results.jsonl").open("a", encoding="utf-8") as results, TraceCapture(test_common, output_dir) as capture:
            for shape, shape_pairs in grouped.items():
                pending = [(pair, batch, order) for pair, batch, order in schedule
                           if shape_key(pair) == shape and identity(pair, batch) not in completed]
                if not pending:
                    continue
                data = reference = None
                try:
                    check_active_gpu(args.gpu, manifest["driver_ownership"])
                    M, N, K = shape
                    data = tuner_module.generate_data(M, N, K, SEED, "128x128", device="cuda:0")
                    reference = tuner_module.run_torch(data["x_deq"], data["w_deq"], torch.bfloat16)
                    torch.cuda.synchronize()
                    label = shape_label(shape)
                    rotation = manifest["shape_rotations"].get(label)
                    if rotation is None:
                        rotation = ({"policy": "explicit-fixed", "num_rotate_args": args.rotate_args,
                                     "probes": [], "recorded_utc": timestamp()} if args.rotate_args else
                                    probe_shared_rotation(torch, test_common, tuner_module, data, shape_pairs))
                        manifest["shape_rotations"][label] = rotation
                        atomic_json(manifest_path, manifest)
                    rotation_args = rotation["num_rotate_args"]
                    if type(rotation_args) is not int or not 1 <= rotation_args <= ITERS:
                        raise RecheckError("Invalid frozen rotation count")

                    for pair, batch, order in pending:
                        row = row_base(pair, batch, order)
                        row.update(rotation_args=rotation_args, started_utc=timestamp(), status="error",
                                   us=None, errRatio=None, trace_path=None, raw_iteration_us=[])
                        capture.active, capture.records = dict(row), []
                        try:
                            row["occupancy_check"] = check_active_gpu(args.gpu, manifest["driver_ownership"])
                            arguments = tuple(data[field] for field in ("x", "weight_shuffle", "x_scale", "w_scale", "out")) + (int(pair["kernelId"]),)
                            info = (("gfx950", 256, *shape, "128x128"), int(pair["kernelId"]),
                                    1, pair["kernelName"], "opus")
                            _, latency, error_ratio = worker(
                                gpu_id=0, info=info, func=tuner_module.run_gemm_opus,
                                args=arguments,
                                kwargs={"num_iters": ITERS, "num_warmup": WARMUP,
                                        "num_rotate_args": rotation_args},
                                ref=reference, rtol=0.01, atol=0.01, printLog=False,
                                tol_err_ratio=ERR_LIMIT, output_keys=("out",),
                                _arg_key_list=["x", "weight_shuffle", "x_scale", "w_scale", "out"],
                                catastrophic_check=True)
                            latency, error_ratio = finite_number(latency), finite_number(error_ratio)
                            row["errRatio"] = error_ratio
                            if latency is not None and latency > 0 and error_ratio is not None and 0 <= error_ratio <= ERR_LIMIT:
                                row.update(status="ok", us=latency)
                            else:
                                row.update(measured_us=latency, error="Benchmark/correctness failure returned by the original worker")
                        except Exception as error:
                            row["error"] = f"{type(error).__name__}: {error}"
                        finally:
                            capture.active = None
                        if capture.records:
                            raw = capture.records[-1]
                            row["trace_path"] = raw["trace_path"]
                            row["raw_iteration_us"] = raw["raw_iteration_us"]
                            samples = raw["raw_iteration_us"]
                            row["timing_details"] = {
                                "profiler_records": len(capture.records),
                                "trace_capture_errors": [record["capture_error"] for record in capture.records if record["capture_error"]],
                                "grouping_note": raw.get("grouping_note"),
                                "raw_calls": len(samples),
                                "raw_median_us": statistics.median(samples) if samples else None,
                                "raw_min_us": min(samples) if samples else None,
                                "raw_max_us": max(samples) if samples else None,
                                "original_get_trace_perf_unchanged": True,
                            }
                        row["completed_utc"] = timestamp()
                        append_row(results, row)
                        if row["status"] != "ok":
                            raise RecheckError(f"Recorded failed row for shape {shape}, kid {pair['kernelId']}, batch {batch}")
                        completed[identity(pair, batch)] = row
                        manifest["completed_results"] = len(completed)
                        atomic_json(manifest_path, manifest)
                        print(json.dumps({"shape": list(shape), "kernelId": int(pair["kernelId"]),
                                          "batch": batch, "us": row["us"], "errRatio": row["errRatio"],
                                          "completed_results": len(completed)}, allow_nan=False), flush=True)
                finally:
                    data = reference = None
                    # The call-local argument tuple also owns tensors until cleared.
                    arguments = None
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
        manifest.update(status="complete", completed_results=len(completed), finished_utc=timestamp())
        atomic_json(manifest_path, manifest)
        return 0
    except Exception as error:
        manifest.update(status="error", error=f"{type(error).__name__}: {error}",
                        completed_results=len(completed), failed_utc=timestamp())
        atomic_json(manifest_path, manifest)
        raise
    finally:
        # Includes both SOs, CO files, sidecar, receipt, and frozen input artifacts.
        try:
            verify_fingerprints(checked["fingerprints"])
            manifest["post_fingerprints_verified"] = True
        except Exception as error:
            manifest.update(status="error", post_fingerprints_verified=False,
                            fingerprint_error=f"{type(error).__name__}: {error}")
            atomic_json(manifest_path, manifest)
            raise
        atomic_json(manifest_path, manifest)


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", choices=("plan", "check", "execute"))
    aliases = parser.add_mutually_exclusive_group()
    aliases.add_argument("--plan", dest="action_alias", action="store_const", const="plan")
    aliases.add_argument("--check", dest="action_alias", action="store_const", const="check")
    aliases.add_argument("--dry-run", dest="action_alias", action="store_const", const="plan")
    aliases.add_argument("--execute", dest="action_alias", action="store_const", const="execute")
    parser.add_argument("--scope", choices=("all", "priority", "remaining"), default="priority")
    parser.add_argument("--execute-gpu", action="store_true", help="Additional explicit gate for the execute action")
    parser.add_argument("--gpu", type=int, choices=(4, 5, 6, 7), help="Explicit physical GPU; never auto-select")
    parser.add_argument("--rotate-args", type=int, help="Fixed shared rotation count, 1..101; default shared-auto per shape")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.action_alias is not None:
        if args.action is not None and args.action != args.action_alias:
            parser.error("Conflicting action and action alias")
        args.action = args.action_alias
    args.action = args.action or "plan"
    if args.rotate_args is not None and not 1 <= args.rotate_args <= ITERS:
        parser.error("--rotate-args must be between 1 and 101")
    if args.action == "execute" and (not args.execute_gpu or args.gpu is None):
        parser.error("GPU execution requires execute --execute-gpu --gpu {4,5,6,7}; no GPU imports performed")
    if args.action != "execute" and args.execute_gpu:
        parser.error("--execute-gpu is only accepted with the execute action; no GPU imports performed")
    if args.action != "execute" and args.resume:
        parser.error("--resume is only accepted with the execute action")
    return args


def main(argv=None):
    args = parse_arguments(argv)
    try:
        preflight_module, checked, pairs, shape_order = load_preflight(args.scope)
        if args.action in ("plan", "check"):
            print(json.dumps(describe_plan(checked, pairs, shape_order, args),
                             indent=2, ensure_ascii=False, allow_nan=False))
            return 0
        return execute_gpu(preflight_module, checked, pairs, shape_order, args)
    except Exception as error:
        print(json.dumps({"status": "error", "error": f"{type(error).__name__}: {error}",
                          "action": args.action, "scope": args.scope}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
