"""Check the integrated exact-kid launchers, tails and scale-panel boundaries."""
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
os.environ.update(OPUS_TUNE_GPU_INDEX="7")
from benchmark import GPU_UUID, GPU_BUS, check_output, require_idle, snapshot
from bootstrap import load_tune

for key in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
    os.environ.pop(key, None)
os.environ.update(ROCR_VISIBLE_DEVICES=GPU_UUID, AITER_AOT_IMPORT="1", AITER_REBUILD="0",
                  AITER_JIT_DIR=str(HERE / "jit"), GPU_ARCHS="gfx950", CU_NUM="256",
                  OMP_NUM_THREADS="2")
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
state = dict(status="running", gpu=7, gpu_uuid=GPU_UUID, start_time=time.time(),
             checks=[], source_sha256=before[0], binary_sha256=before[1])
class Telemetry:
    def sample(self, row):
        with (HERE / "validation_gpu.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
telemetry = Telemetry()
require_idle(torch, telemetry, "validation_start")
edges = [(m, 256, k) for m in (64, 128, 192, 256, 320) for k in (128, 256, 384)]
edges += [(m, 512, k) for m in (64, 192, 256, 1088)
          for k in (8064, 8192, 8320, 16256, 16384, 16512)]
try:
    for m, n, k in edges:
        data = tune.generate_data(m, n, k, device="cuda:0")
        reference = tune.run_torch(*(data[key] for key in tune._REF_KEYS), with_bounds=True)
        storage = torch.full((m + 128, n), float("nan"), device="cuda", dtype=torch.bfloat16)
        out = storage[64:64 + m]
        assert out.is_contiguous()
        for kid in range(9030, 9034):
            storage.fill_(float("nan"))
            result = opus_gemm(data["x"], data["w"], out, kid=kid, layout="bpreshuffle",
                               x_scale=data["x_scale"], w_scale=data["w_scale"])
            assert result is out
            checked = check_output(torch, tune, reference, out)
            guards = bool(torch.isnan(storage[:64]).all() and torch.isnan(storage[64+m:]).all())
            row = dict(M=m, N=n, K=k, kid=kid, guards=guards, **checked)
            state["checks"].append(row)
            assert checked["status"] == "passed" and guards, row
        del data, reference, storage, out, result
    for m, n, k in ((65, 256, 128), (64, 128, 128), (64, 256, 192)):
        x = torch.empty((m, k), device="cuda", dtype=dtypes.fp8)
        w = torch.empty((n, k), device="cuda", dtype=dtypes.fp8)
        out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
        sa = torch.empty((max(1, k//128), m), device="cuda", dtype=torch.uint8).T
        sb = torch.empty((n//128, max(1, k//128)), device="cuda", dtype=torch.uint8)
        for kid in range(9030, 9034):
            try:
                opus_gemm(x, w, out, kid=kid, layout="bpreshuffle", x_scale=sa, w_scale=sb)
            except RuntimeError as exc:
                assert "multiple" in str(exc), str(exc)
                state["checks"].append(dict(M=m, N=n, K=k, kid=kid, status="rejected_invalid_alignment"))
            else:
                raise AssertionError(f"Invalid shape accepted: {(m,n,k,kid)}")
    require_idle(torch, telemetry, "validation_end")
    assert snapshot(HERE / "jit") == before
    state.update(status="passed", boundary_checks=156, rejected_invalid_shapes=12)
except BaseException as exc:
    state.update(status="failed", error=repr(exc))
    raise
finally:
    state["end_time"] = time.time()
    (HERE / "validation.json").write_text(json.dumps(state, indent=2) + "\n")
    print(json.dumps({k:v for k,v in state.items() if k not in ("source_sha256", "binary_sha256", "checks")}))
