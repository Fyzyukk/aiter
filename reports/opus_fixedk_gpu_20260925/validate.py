"""GPU numerical/guard/rejection validation for both independent fixed-K cohorts."""
import csv
import json
import os
from pathlib import Path
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
os.environ["OPUS_TUNE_GPU_INDEX"] = "7"
from benchmark import GPU_UUID, GPU_BUS, check_output, require_idle, snapshot
from bootstrap import load_tune

for key in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
    os.environ.pop(key, None)
os.environ.update(ROCR_VISIBLE_DEVICES=GPU_UUID, AITER_AOT_IMPORT="1", AITER_REBUILD="0",
                  AITER_JIT_DIR=str(HERE / "jit"), GPU_ARCHS="gfx950", CU_NUM="256",
                  OMP_NUM_THREADS="2", AITER_LOG_MORE="0", AITER_SMI_MONITOR="0")
sys.path.insert(0, str(ROOT))
import torch
import aiter
from aiter.utility import dtypes
aiter.dtypes = dtypes
from aiter.jit import core
from aiter.ops.opus import opus_gemm


def forbid_build(*args, **kwargs):
    raise RuntimeError("Validation requires prebuilt modules")


core.build_module = forbid_build
tune = load_tune()
torch.set_float32_matmul_precision("highest")
torch.backends.cuda.matmul.allow_tf32 = False
assert torch.cuda.device_count() == 1
prop = torch.cuda.get_device_properties(0)
assert prop.pci_bus_id == GPU_BUS and prop.multi_processor_count == 256
assert prop.gcnArchName.startswith("gfx950")
before = snapshot(HERE / "jit")
result_path = HERE / "validation.json"
assert not result_path.exists()
state = dict(status="running", gpu=7, gpu_uuid=GPU_UUID, start_time=time.time(),
             checks=[], source_sha256=before[0], binary_sha256=before[1],
             contract="Original FP32 accumulation bounds; every element error==0; NaN output/guards; no relaxed tolerances")


def save():
    temp = result_path.with_suffix(".tmp")
    temp.write_text(json.dumps(state, indent=2) + "\n")
    temp.replace(result_path)


class Telemetry:
    def sample(self, row):
        with (HERE / "validation_gpu.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")


telemetry = Telemetry()
kids_by_k = {384: (9040,), 768: (9041,), 1024: (9042,), 1536: (9050, 9051)}
targets = [tuple(int(row[key]) for key in ("M", "N", "K"))
           for row in csv.DictReader((HERE / "shapes33.csv").open())]
assert len(targets) == len(set(targets)) == 33
edges = [(m, n, k) for k in kids_by_k for m in (64, 128, 192, 256, 320)
         for n in (256, 512)]
save()
try:
    require_idle(torch, telemetry, "validation_start")
    for index, (m, n, k) in enumerate(targets + edges):
        data = tune.generate_data(m, n, k, device="cuda:0")
        reference = tune.run_torch(*(data[key] for key in tune._REF_KEYS), with_bounds=True)
        storage = torch.full((m + 128, n), float("nan"), device="cuda", dtype=torch.bfloat16)
        out = storage[64:64 + m]
        assert out.is_contiguous()
        for kid in kids_by_k[k]:
            storage.fill_(float("nan"))
            result = opus_gemm(data["x"], data["w"], out, kid=kid, layout="bpreshuffle",
                               x_scale=data["x_scale"], w_scale=data["w_scale"])
            assert result is out
            checked = check_output(torch, tune, reference, out)
            guards = bool(torch.isnan(storage[:64]).all() and torch.isnan(storage[64+m:]).all())
            row = dict(M=m, N=n, K=k, kid=kid, guards=guards, phase="target" if index < 33 else "edge", **checked)
            state["checks"].append(row)
            save()
            assert checked["status"] == "passed" and guards, row
            if (m, n) == (64, 256):
                storage.fill_(float("nan"))
                result = opus_gemm(data["x"], data["w"], out, kid=kid, layout="bpreshuffle",
                                   x_scale=data["x_scale"].view(torch.uint8),
                                   w_scale=data["w_scale"].view(torch.uint8))
                checked = check_output(torch, tune, reference, out)
                guards = bool(torch.isnan(storage[:64]).all() and torch.isnan(storage[64+m:]).all())
                row = dict(M=m, N=n, K=k, kid=kid, guards=guards, phase="raw_e8m0", **checked)
                state["checks"].append(row)
                assert checked["status"] == "passed" and guards, row
        print(json.dumps({"shape": [m,n,k], "kids": kids_by_k[k], "status": "passed"}), flush=True)
        del data, reference, storage, out, result
    for fixed_k, kids in kids_by_k.items():
        invalid = [(65, 256, fixed_k), (64, 128, fixed_k), (64, 256, fixed_k + 1)]
        invalid += [(64, 256, other) for other in (128,384,768,1024,1536,3072) if other != fixed_k]
        for m, n, k in invalid:
            x = torch.empty((m,k), device="cuda", dtype=dtypes.fp8)
            w = torch.empty((n,k), device="cuda", dtype=dtypes.fp8)
            out = torch.empty((m,n), device="cuda", dtype=torch.bfloat16)
            sa = torch.empty((max(1,k//128),m), device="cuda", dtype=torch.uint8).T
            sb = torch.empty((n//128,max(1,k//128)), device="cuda", dtype=torch.uint8)
            for kid in kids:
                try:
                    opus_gemm(x,w,out,kid=kid,layout="bpreshuffle",x_scale=sa,w_scale=sb)
                except RuntimeError as exc:
                    assert "multiple" in str(exc) or f"requires K == {fixed_k}" in str(exc), str(exc)
                    state["checks"].append(dict(M=m,N=n,K=k,kid=kid,status="rejected_invalid_shape",reason=str(exc)))
                else:
                    raise AssertionError(f"Invalid shape accepted: {(m,n,k,kid)}")
    require_idle(torch, telemetry, "validation_end")
    assert snapshot(HERE / "jit") == before
    state.update(status="passed", numerical_checks=sum(r["status"] == "passed" for r in state["checks"]),
                 rejected_invalid_shapes=sum(r["status"] == "rejected_invalid_shape" for r in state["checks"]))
except BaseException as exc:
    state.update(status="failed",error=repr(exc),traceback=traceback.format_exc())
    raise
finally:
    state["end_time"] = time.time()
    save()
    print(json.dumps({k:v for k,v in state.items() if k not in ("source_sha256","binary_sha256","checks")}))
