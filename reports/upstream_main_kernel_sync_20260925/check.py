"""Build the five retained kernels and check integration on current upstream."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MODE = sys.argv[1]
JIT = HERE / "jit"
JIT.mkdir(exist_ok=True)
for key in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
    os.environ.pop(key, None)
os.environ.update(
    ROCR_VISIBLE_DEVICES="GPU-5ff36708541c8ec0",
    AITER_AOT_IMPORT="1", AITER_REBUILD="0", AITER_JIT_DIR=str(JIT),
    GPU_ARCHS="gfx950", CU_NUM="256", OMP_NUM_THREADS="2", MAX_JOBS="12",
    HIP_CLANG_PATH="/root/toolchains/llvm-amdgpu-pin-op-dst-49c41889-build/bin",
)
sys.path[:0] = [str(ROOT), str(ROOT / "csrc/opus_gemm"),
                str(ROOT / "csrc/ck_gemm_a8w8_blockscale")]
import aiter
from aiter.utility import dtypes
aiter.dtypes = dtypes
import torch

if MODE == "build":
    from opus_gemm_tune import _ensure_kids_compiled
    _ensure_kids_compiled({9000, 9010, 9011, 9012, 9020})
    result = dict(status="passed", kids=[9000, 9010, 9011, 9012, 9020],
                  binary_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in JIT.glob("*.so")})
    (HERE / "build.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    raise SystemExit(0)

from aiter.jit import core
def forbid_build(*args, **kwargs):
    raise RuntimeError("Tests must use prebuilt modules")
core.build_module = forbid_build
import pytest
if MODE == "cpu":
    files = [ROOT / "op_tests" / name for name in (
        "test_opus_a8w8_interface.py", "test_opus_a16w16_policy_parity.py",
        "test_opus_co_integration.py",
    )]
    arguments = list(map(str, files))
elif MODE == "gpu":
    assert torch.cuda.device_count() == 1
    assert torch.cuda.get_device_properties(0).pci_bus_id == 0xF5
    name = "csrc.opus_gemm.opus_gemm_mxscale_bpreshuffle_tune"
    spec = importlib.util.spec_from_file_location(name, HERE / "opus_gemm_mxscale_bpreshuffle_tune.py")
    tune = importlib.util.module_from_spec(spec)
    sys.modules[name] = tune
    spec.loader.exec_module(tune)
    arguments = [str(HERE / "test_opus_mxscale_bpreshuffle_tune.py"), "-k",
                 "test_gpu or test_candidates_cover_all_registered_implementations or test_bpreshuffle_codegen"]
else:
    raise ValueError(MODE)
xml = HERE / f"{MODE}.xml"
code = pytest.main([*arguments, "-q", "-x", "--junitxml=" + str(xml)])
cases = ET.parse(xml).getroot().findall(".//testcase")
summary = dict(status="passed" if code == 0 else "failed", cases=len(cases),
               skipped=sum(c.find("skipped") is not None for c in cases), returncode=int(code))
(HERE / f"{MODE}.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary), flush=True)
raise SystemExit(code)
