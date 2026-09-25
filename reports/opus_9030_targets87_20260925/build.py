"""Build all measured modules from the current checkout into a fresh JIT cache."""
import hashlib
import inspect
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
JIT = HERE / "jit"
STOCK = "/opt/rocm/llvm/bin"
OPUS = "/root/toolchains/llvm-amdgpu-pin-op-dst-49c41889-build/bin"
MODULES = ["module_aiter_core", "module_gemm_a8w8_blockscale_bpreshuffle_tune",
           "module_gemm_a8w8_blockscale_bpreshuffle_cktile_tune",
           "module_gemm_a8w8_blockscale_bpreshuffle_asm", "module_deepgemm_opus"]

if len(sys.argv) > 1:
    name = sys.argv[1]
    assert name in MODULES
    sys.path[:0] = [str(ROOT), str(ROOT / "csrc/opus_gemm")]
    import aiter
    from aiter.utility import dtypes
    aiter.dtypes = dtypes
    if name == "module_deepgemm_opus":
        from opus_gemm_tune import _ensure_kids_compiled
        _ensure_kids_compiled({9000, 9010, 9011, 9012, 9020, 9030, 9031, 9032, 9033})
    else:
        from aiter.jit import core
        config = core.get_args_of_build(name)
        config["md_name"] = name
        allowed = inspect.signature(core.build_module).parameters
        core.build_module(**{k: v for k, v in config.items() if k in allowed})
    assert (JIT / (name + ".so")).is_file()
    print("Built", name, flush=True)
    raise SystemExit(0)

assert not (HERE / "build.json").exists()
JIT.mkdir(exist_ok=False)
state = dict(status="running", start_time=time.time(),
             head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
             submodules=subprocess.check_output(["git", "submodule", "status"], cwd=ROOT, text=True).strip(),
             jit_directory=str(JIT), modules=[], compilers={})
for label, path in (("ck_cktile_asm", STOCK), ("opus", OPUS)):
    state["compilers"][label] = subprocess.check_output([path + "/clang++", "--version"], text=True)
def save():
    (HERE / "build.json").write_text(json.dumps(state, indent=2) + "\n")
save()
try:
    previous = ROOT / "reports/opus_local_gap_current_20260925"
    prior_build = json.loads((previous / "build.json").read_text())
    prior_run = json.loads((previous / "full_r3_run.json").read_text())
    assert prior_build["status"] == prior_run["status"] == "passed"
    for path, digest in prior_run["reference_sha256"].items():
        if path.startswith(("jit/", "asm/")):
            continue
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest, path
    state["reference_reuse"] = {"build": str(previous / "build.json"),
                                "run": str(previous / "full_r3_run.json"),
                                "source_hashes_verified": True}
    save()
    for name in MODULES:
        if name != "module_deepgemm_opus":
            source = previous / "jit" / (name + ".so")
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            assert digest == prior_build["binary_sha256"][source.name]
            shutil.copyfile(source, JIT / source.name)
            state["modules"].append(dict(name=name, exit_code=0, reused_from=str(source), sha256=digest))
            save()
            continue
        env = os.environ.copy()
        for key in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "AITER_LOG_MORE", "AITER_SMI_MONITOR"):
            env.pop(key, None)
        env.update(ROCR_VISIBLE_DEVICES="GPU-5ff36708541c8ec0", AITER_AOT_IMPORT="1",
                   AITER_REBUILD="0", AITER_JIT_DIR=str(JIT), GPU_ARCHS="gfx950", CU_NUM="256",
                   OMP_NUM_THREADS="2", MAX_JOBS="12",
                   HIP_CLANG_PATH=OPUS if name == "module_deepgemm_opus" else STOCK)
        state["building"] = name
        save()
        started = time.time()
        with (HERE / (name + "_build.log")).open("x") as log:
            result = subprocess.run([sys.executable, "-u", __file__, name], cwd=ROOT, env=env,
                                    stdout=log, stderr=subprocess.STDOUT)
        state["modules"].append(dict(name=name, exit_code=result.returncode, seconds=time.time()-started))
        save()
        assert result.returncode == 0, name
        print("Built", name, flush=True)
    state.update(status="passed", binary_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                                  for p in JIT.glob("*.so")})
except BaseException as exc:
    state.update(status="failed", error=repr(exc))
    raise
finally:
    state["end_time"] = time.time()
    save()
