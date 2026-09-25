"""Isolate timing settings using the frozen CKTile binary from the full sweep.

This diagnoses the old report; it is not acceptance of the new upstream build.
"""
import csv
import hashlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FROZEN = ROOT / "reports/opus_mxfp8_full_retune_20260925"
JIT = FROZEN / "jit_03"
for key in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
    os.environ.pop(key, None)
os.environ.update(
    ROCR_VISIBLE_DEVICES="GPU-56f0ab624008ec65",
    AITER_JIT_DIR=str(JIT), AITER_AOT_IMPORT="1", AITER_REBUILD="0",
    GPU_ARCHS="gfx950", CU_NUM="256", OMP_NUM_THREADS="2", AITER_LOG_MORE="0",
)
sys.path[:0] = [str(ROOT), str(ROOT / "csrc/ck_gemm_a8w8_blockscale")]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

manifest = json.loads((FROZEN / "jit_03_build.json").read_text())
binary = JIT / "module_gemm_a8w8_blockscale_bpreshuffle_cktile_tune.so"
assert sha(binary) == manifest["binary_sha256"][binary.name]

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

import aiter
from aiter.utility import dtypes
aiter.dtypes = dtypes
from aiter.jit import core
def forbid_build(*args, **kwargs):
    raise RuntimeError("Diagnostic must not build or replace frozen modules")
core.build_module = forbid_build
from aiter.ops import gemm_op_a8w8
aiter.gemm_a8w8_blockscale_bpreshuffle_cktile_tune = gemm_op_a8w8.gemm_a8w8_blockscale_bpreshuffle_cktile_tune
import torch
aiter.logger.setLevel(logging.WARNING)
assert torch.cuda.device_count() == 1
props = torch.cuda.get_device_properties(0)
assert props.pci_bus_id == 0xE5
torch.set_float32_matmul_precision("highest")
torch.backends.cuda.matmul.allow_tf32 = False

tune_rel = "csrc/opus_gemm/opus_gemm_mxscale_bpreshuffle_tune.py"
perf_rel = "aiter/test_common.py"
for rel in (tune_rel, perf_rel):
    assert sha(FROZEN / "jit_03_source" / rel) == manifest["source_sha256"][rel]
tune = load("diagnosis_frozen_tune", FROZEN / "jit_03_source" / tune_rel)
perf = load("diagnosis_frozen_perf", FROZEN / "jit_03_source" / perf_rel)

cases = [(4096, 2048, 7168, 27), (1280, 7168, 7168, 27), (6144, 6144, 7168, 28)]
protocols = [("rotate_5_51", 0, 5, 51), ("reuse_5_51", 1, 5, 51),
             ("rotate_50_200", 0, 50, 200), ("reuse_50_200", 1, 50, 200)]
result = dict(
    status="running", purpose=__doc__, gpu=6, gpu_properties=str(props),
    torch_version=torch.__version__, hip_version=torch.version.hip,
    binary=str(binary), binary_sha256=sha(binary), source_sha256={
        rel: manifest["source_sha256"][rel] for rel in (tune_rel, perf_rel)},
    records=[], summaries=[], idle_checks=[], start_time=time.time(),
)
output = HERE / (sys.argv[1] if len(sys.argv) > 1 else "cache_probe.json")
if output.exists():
    raise FileExistsError(output)
output.write_text(json.dumps(result, indent=2) + "\n")

def save():
    output.write_text(json.dumps(result, indent=2) + "\n")

def require_idle():
    torch.cuda.synchronize()
    recent = []
    for _ in range(20):
        time.sleep(0.4)
        state = json.loads(subprocess.check_output(
            ["rocm-smi", "-d", "6", "--showuse", "--showmemuse", "--json"], text=True))["card6"]
        recent.append(state)
        result["idle_checks"].append(dict(time=time.time(), **state))
        window = recent[-3:]
        if (len(window) == 3 and not any(int(s["GPU use (%)"]) for s in window)
                and len({s["GFX Activity"] for s in window}) == 1):
            return
    raise RuntimeError("GPU did not become idle between shapes")

try:
    for m, n, k, kid in cases:
        require_idle()
        data = tune.generate_data(m, n, k, device="cuda:0")
        ref = tune.run_torch(*(data[key] for key in tune._REF_KEYS), with_bounds=True)
        args = tuple(data[key] for key in tune.cktile_bench_keys(kid)) + (kid, 0, True)
        func = tune.generic_tune.run_gemm_a8w8_blockscale_cktile
        data["out"].fill_(float("nan"))
        out = func(*args)
        assert tune.compare_outputs(ref, out, printLog=False) == 0
        for turn in range(3):
            ordered = protocols[turn:] + protocols[:turn]
            for name, rotation, warmup, iters in ordered:
                out, us = perf.run_perftest(func, *args, num_rotate_args=rotation,
                                           num_warmup=warmup, num_iters=iters)
                err = tune.compare_outputs(ref, out, printLog=False)
                assert err == 0, (m, n, k, kid, name, err)
                row = dict(M=m, N=n, K=k, kid=kid, protocol=name, round=turn,
                           num_rotate_args=rotation, warmup=warmup, iters=iters,
                           us=float(us), err=err)
                result["records"].append(row)
                save()
        for name, rotation, warmup, iters in protocols:
            values = [r["us"] for r in result["records"]
                      if (r["M"], r["N"], r["K"], r["protocol"]) == (m, n, k, name)]
            row = dict(M=m, N=n, K=k, kid=kid, protocol=name,
                       median_us=statistics.median(values), samples=values)
            result["summaries"].append(row)
            print(json.dumps(row), flush=True)
        del args, data, ref, out
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    assert sha(binary) == result["binary_sha256"]
    result["status"] = "passed"
except BaseException as exc:
    result.update(status="failed", error=repr(exc))
    raise
finally:
    result["end_time"] = time.time()
    save()
