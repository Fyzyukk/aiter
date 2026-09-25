"""Build current registered OPUS candidates in a new, isolated JIT directory."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
name = sys.argv[1] if len(sys.argv) > 1 else "jit_01"
assert Path(name).name == name
JIT = HERE / name
JIT.mkdir(exist_ok=False)
old = ROOT / "reports/opus_9010_9011_opt_20260924"
previous = json.loads((old / "harness/final_r5_run.json").read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
for path, expected in previous["reference_sha256"].items():
    source = ROOT / path
    if path.startswith("asm/"):
        source = ROOT / "hsa/gfx950/fp8gemm_blockscale" / Path(path).name
    elif path.startswith("jit/"):
        source = old / "jit_final" / Path(path).name
    assert sha(source) == expected, path
for source in (old / "jit_final").glob("*.so"):
    if source.name != "module_deepgemm_opus.so":
        expected = previous["binary_sha256"].get(str(source))
        if expected is not None:
            assert sha(source) == expected, source
        shutil.copy2(source, JIT / source.name)
for key in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
    os.environ.pop(key, None)
os.environ.update(
    ROCR_VISIBLE_DEVICES="GPU-5ff36708541c8ec0", AITER_AOT_IMPORT="1",
    AITER_REBUILD="0", AITER_JIT_DIR=str(JIT), GPU_ARCHS="gfx950", CU_NUM="256",
    MAX_JOBS="12", OMP_NUM_THREADS="2",
    HIP_CLANG_PATH="/root/toolchains/llvm-amdgpu-pin-op-dst-49c41889-build/bin",
)
sys.path.insert(0, str(ROOT))
import aiter
from aiter.utility import dtypes
aiter.dtypes = dtypes
from csrc.opus_gemm import opus_gemm_mxscale_bpreshuffle_tune as tune

paths = set(previous["source_sha256"])
paths.update(str(p.relative_to(ROOT)) for p in (ROOT / "csrc/opus_gemm/include/gfx950").glob("*mxscale_bpreshuffle*"))
paths.update(["reports/opus_mxfp8_full_retune_20260925/build.py",
              "reports/opus_mxfp8_padded_m_20260922/joint_tune_worker.py"])
hashes = {p: sha(ROOT / p) for p in sorted(paths)}
for path in hashes:
    target = HERE / (name + "_source") / path
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / path, target)
manifest = dict(status="building", jit_directory=str(JIT), source_sha256=hashes,
                opus_kids=[9000, 9010, 9011, 9012, 9020],
                configuration={"9010": "S3/K+2", "9011": "S3/K+2", "9012": "S4/K+3, direct BF16"})
output = HERE / (name + "_build.json")
output.write_text(json.dumps(manifest, indent=2) + "\n")
try:
    tune._ensure_kids_compiled(set(manifest["opus_kids"]))
    assert all(sha(ROOT / p) == h for p, h in hashes.items())
    manifest.update(status="passed", binary_sha256={p.name: sha(p) for p in JIT.glob("*.so")})
except BaseException as exc:
    manifest.update(status="failed", error=repr(exc))
    raise
finally:
    output.write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps({k: v for k, v in manifest.items() if k != "source_sha256"}, indent=2), flush=True)
