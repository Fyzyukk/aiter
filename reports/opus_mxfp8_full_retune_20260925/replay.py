"""Replay every exported all-backend winner through the native E8M0 tuner."""
import hashlib
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
run = json.loads((HERE / "full_r3_run.json").read_text())
assert run["status"] == "passed"
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check_hashes():
    for p, h in run["source_sha256"].items():
        assert sha(ROOT / p) == h, p
    for p, h in run["binary_sha256"].items():
        assert sha(p) == h, p
check_hashes()
for key in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
    os.environ.pop(key, None)
os.environ.update(ROCR_VISIBLE_DEVICES=run["gpu_uuid"], AITER_AOT_IMPORT="1",
                  AITER_REBUILD="0", AITER_JIT_DIR=run["jit_dir"], GPU_ARCHS="gfx950",
                  CU_NUM="256", OMP_NUM_THREADS="2")
sys.path.insert(0, str(ROOT))
import aiter
from aiter.utility import dtypes
aiter.dtypes = dtypes
from aiter.ops import gemm_op_a8w8
for name in ("gemm_a8w8_blockscale_bpreshuffle_tune", "gemm_a8w8_blockscale_bpreshuffle_cktile_tune",
             "gemm_a8w8_blockscale_bpreshuffle_asm"):
    setattr(aiter, name, getattr(gemm_op_a8w8, name))
from aiter.jit import core
def forbid_build(*args, **kwargs):
    raise RuntimeError("Replay must use the measured module")
core.build_module = forbid_build
import torch
assert torch.cuda.device_count() == 1
assert torch.cuda.get_device_properties(0).pci_bus_id == 0xF5
from csrc.opus_gemm.opus_gemm_mxscale_bpreshuffle_tune import OpusMxscaleBpreshuffleTuner
tuner = OpusMxscaleBpreshuffleTuner()
args = tuner.parser.parse_args(["--run_config", str(HERE / "tuned.csv"), "--mp", "1",
                               "--warmup", "5", "--iters", "51", "--errRatio", "0"])
tuner.pre_process(args)
results = tuner.run_config(args)
(HERE / "replay_rows.json").write_text(json.dumps(results, indent=2) + "\n")
assert len(results) == 295
assert all(r["status"] == "ok" and r["errRatio"] == 0 for r in results)
torch.cuda.synchronize()
check_hashes()
summary = dict(status="passed", shapes=len(results), tuned_sha256=sha(HERE / "tuned.csv"),
               measured_binary_sha256=run["binary_sha256"], sources_unchanged=True)
(HERE / "replay.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps({"status": "passed", "replayed_shapes": len(results)}), flush=True)
