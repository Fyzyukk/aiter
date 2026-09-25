"""Replay every exported all-backend winner through the native E8M0 tuner."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, choices=range(4, 8), required=True)
parser.add_argument('--config', type=Path, required=True)
args_cli = parser.parse_args()
GPU_MAP = {4: ('GPU-23a6cd0d658d72b6', 0x85), 5: ('GPU-2d57f9bd7c2ee0fe', 0x95),
           6: ('GPU-56f0ab624008ec65', 0xE5), 7: ('GPU-5ff36708541c8ec0', 0xF5)}
GPU_UUID, GPU_BUS = GPU_MAP[args_cli.gpu]
config = args_cli.config.resolve()
with config.open() as f:
    expected_rows = list(csv.DictReader(f))
assert expected_rows
prefix = HERE / f'gpu{args_cli.gpu}_replay'
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
os.environ.update(ROCR_VISIBLE_DEVICES=GPU_UUID, AITER_AOT_IMPORT="1",
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
assert torch.cuda.get_device_properties(0).pci_bus_id == GPU_BUS
from bootstrap import load_tune
OpusMxscaleBpreshuffleTuner = load_tune().OpusMxscaleBpreshuffleTuner
tuner = OpusMxscaleBpreshuffleTuner()
args = tuner.parser.parse_args(["--run_config", str(config), "--mp", "1",
                               "--warmup", "5", "--iters", "51", "--errRatio", "0"])
tuner.pre_process(args)
results = tuner.run_config(args)
Path(str(prefix) + "_rows.json").write_text(json.dumps(results, indent=2) + "\n")
assert len(results) == len(expected_rows)
assert all(r["status"] == "ok" and r["errRatio"] == 0 for r in results)
torch.cuda.synchronize()
check_hashes()
summary = dict(status="passed", shapes=len(results), tuned_sha256=sha(config), gpu=args_cli.gpu, gpu_uuid=GPU_UUID,
               measured_binary_sha256=run["binary_sha256"], sources_unchanged=True)
Path(str(prefix) + ".json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps({"status": "passed", "replayed_shapes": len(results)}), flush=True)
