"""Run the existing full MXFP8 test file against this run's frozen module."""
import hashlib
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
name = sys.argv[1] if len(sys.argv) > 1 else "jit_03"
build = json.loads((HERE / (name + "_build.json")).read_text())
assert build["status"] == "passed"
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check_hashes():
    for p, h in build["source_sha256"].items():
        assert sha(ROOT / p) == h, p
    for p, h in build["binary_sha256"].items():
        assert sha(HERE / name / p) == h, p
check_hashes()
for key in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
    os.environ.pop(key, None)
os.environ.update(ROCR_VISIBLE_DEVICES="GPU-5ff36708541c8ec0", AITER_AOT_IMPORT="1",
                  AITER_REBUILD="0", AITER_JIT_DIR=str(HERE / name), GPU_ARCHS="gfx950",
                  CU_NUM="256", OMP_NUM_THREADS="2")
sys.path.insert(0, str(ROOT))
import aiter
from aiter.utility import dtypes
aiter.dtypes = dtypes
from aiter.jit import core
def forbid_build(*args, **kwargs):
    raise RuntimeError("Validation must use the frozen module")
core.build_module = forbid_build
import torch
assert torch.cuda.device_count() == 1
props = torch.cuda.get_device_properties(0)
assert props.pci_bus_id == 0xF5 and props.multi_processor_count == 256
import pytest
xml = HERE / (name + "_tests.xml")
code = pytest.main([str(ROOT / "op_tests/tuning_tests/test_opus_mxscale_bpreshuffle_tune.py"),
                    "-q", "-x", "--junitxml=" + str(xml)])
cases = ET.parse(xml).getroot().findall(".//testcase")
summary = dict(status="passed" if code == 0 else "failed", cases=len(cases),
               gpu_cases=sum("test_gpu_" in c.attrib["name"] for c in cases),
               skipped=sum(c.find("skipped") is not None for c in cases),
               pytest_returncode=int(code), device=str(props))
check_hashes()
(HERE / (name + "_validation.json")).write_text(json.dumps(summary, indent=2) + "\n")
assert summary["skipped"] == 0
print(json.dumps(summary, indent=2), flush=True)
raise SystemExit(code)
