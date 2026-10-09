#!/usr/bin/env python3
"""Build current candidates with a recorded workaround for this host's LLVM.

ROCm's hipcc -E flag probe does not invoke LLVM's backend option parser.
It accepts amdgpu-coerce-illegal-types even though the pinned LLVM rejects it
when compiling. Filter only that independently observed unsupported option.
This changes the build invocation, not repository kernel or JIT source.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

DEFAULT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source-root', type=Path, default=DEFAULT_ROOT)
parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUT)
parser.add_argument('--jit-dir', type=Path)
parser.add_argument('--manifest', type=Path)
parser.add_argument('--pending-candidate', action='store_true',
                    help='isolate a pending source change; hide GPUs and forbid baseline artifact output')
args = parser.parse_args()
ROOT = args.source_root.resolve()
OUT = args.output_dir.resolve()
OUT.mkdir(parents=True, exist_ok=True)
MANIFEST = args.manifest.resolve() if args.manifest else OUT / 'current_build.json'
if args.jit_dir:
    os.environ['AITER_JIT_DIR'] = str(args.jit_dir.resolve())
if args.pending_candidate:
    if ROOT == DEFAULT_ROOT or OUT == DEFAULT_OUT or MANIFEST == DEFAULT_OUT / 'current_build.json':
        parser.error('pending candidate requires a separate source root, output directory and manifest')
    if Path(os.environ.get('AITER_JIT_DIR', OUT / 'jit_baseline')).resolve() == DEFAULT_OUT / 'jit_baseline':
        parser.error('pending candidate must not write the baseline JIT directory')
    for name in ['HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES']:
        os.environ[name] = ''
os.chdir(ROOT)
os.environ.setdefault("AITER_AOT_IMPORT", "1")
os.environ.setdefault("GPU_ARCHS", "gfx950")
os.environ.setdefault("CU_NUM", "256")
os.environ.setdefault("OPUS_HIP_CLANG_PATH", "/root/toolchains/yuyzhang512-amdgpu-pin-op-dst-build/bin")
os.environ["HIP_CLANG_PATH"] = os.environ["OPUS_HIP_CLANG_PATH"]
os.environ.setdefault("AITER_JIT_DIR", str(OUT / "jit_baseline"))
sys.path.insert(0, str(ROOT))

from aiter.jit import core
from csrc.opus_gemm.opus_gemm_common import A8W8_BPRESHUFFLE_TUNING_KIDS
if not args.pending_candidate:
    from csrc.opus_gemm import opus_gemm_mxscale_bpreshuffle_tune as tune
else:
    # Importing the tuner probes runtime gfx to select Python dtypes and can
    # attempt a CUDA allocation in its enumeration workaround. The candidate
    # prebuild uses the same official build arguments and codegen directly.
    tune = None

original_checker = core.hip_flag_checker
unsupported = "-mllvm -amdgpu-coerce-illegal-types=1"

# ROCm 7.0's HIP half header references legacy OCML declarations removed
# from upstream LLVM 24. The isolated opus_bmm.cu probe passed with the
# installed ROCm resource directory while retaining the pinned compiler.
legacy_declarations = "/opt/rocm/lib/llvm/lib/clang/20/include/__clang_hip_libdevice_declares.h"
resource_directory = "/opt/rocm/lib/llvm/lib/clang/20"
original_build = core.build_module

def compatible_build(*args, **kwargs):
    kwargs["flags_extra_hip"] = list(kwargs.get("flags_extra_hip", [])) + [f"-resource-dir={resource_directory}"]
    return original_build(*args, **kwargs)

core.build_module = compatible_build

def checked(flag):
    return False if flag == unsupported else original_checker(flag)

core.hip_flag_checker = checked
started = time.time()
manifest = {
    "source_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    "python": sys.executable,
    "compiler": os.environ["OPUS_HIP_CLANG_PATH"],
    "filtered_backend_option": unsupported,
    "resource_directory": resource_directory,
    "hip_declaration_header": legacy_declarations,
    "hip_declaration_header_sha256": hashlib.sha256(Path(legacy_declarations).read_bytes()).hexdigest(),
    "reason": "Backend flag rejected; installed ROCm headers needed for legacy half declarations. CPU opus_bmm.cu resource-dir probe passed.",
    "source_root": str(ROOT),
    "output_directory": str(OUT),
    "jit_directory": os.environ['AITER_JIT_DIR'],
    "manifest_path": str(MANIFEST),
    "pending_candidate_no_adoption": args.pending_candidate,
    "kernel_source_changed": args.pending_candidate,
    "gpu_execution_requested": False,
    "visibility": {key: os.environ.get(key) for key in ['HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES']},
    "imported_core": str(Path(core.__file__).resolve()),
    "imported_tune": str(Path(tune.__file__).resolve()) if tune else None,
    "build_entry": 'official core.build_module with the same get_args_of_build and --extra_kids' if args.pending_candidate else 'tune._ensure_kids_compiled',
    "source_diff": subprocess.check_output(['git', 'diff', '--binary'], cwd=ROOT, text=True),
    "kids": sorted(A8W8_BPRESHUFFLE_TUNING_KIDS),
}
try:
    if args.pending_candidate:
        core.AITER_REBUILD = 1
        os.environ['AITER_REBUILD'] = '1'
        build_args = core.get_args_of_build('module_deepgemm_opus')
        core.build_module(
            md_name='module_deepgemm_opus', srcs=build_args['srcs'],
            flags_extra_cc=build_args['flags_extra_cc'], flags_extra_hip=build_args['flags_extra_hip'],
            blob_gen_cmd=build_args['blob_gen_cmd'] + ' --extra_kids ' + ' '.join(str(k) for k in sorted(A8W8_BPRESHUFFLE_TUNING_KIDS)),
            extra_include=build_args['extra_include'], extra_ldflags=build_args['extra_ldflags'],
            verbose=build_args.get('verbose', False), is_python_module=build_args.get('is_python_module', True),
            is_standalone=build_args.get('is_standalone', False), torch_exclude=build_args.get('torch_exclude', False),
            third_party=build_args.get('third_party', []), hipify=build_args.get('hipify', False),
            flags_extra_hip_per_source=build_args.get('flags_extra_hip_per_source', {}), build_after_wait=True)
        manifest['rebuilt'] = True
    else:
        manifest["rebuilt"] = tune._ensure_kids_compiled(A8W8_BPRESHUFFLE_TUNING_KIDS)
    artifact = Path(os.environ["AITER_JIT_DIR"]) / "module_deepgemm_opus.so"
    manifest["binary"] = str(artifact)
    manifest["binary_sha256"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
    manifest["status"] = "passed"
except Exception:
    manifest["status"] = "failed"
    raise
finally:
    manifest["seconds"] = time.time() - started
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest), flush=True)
