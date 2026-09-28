#!/usr/bin/env python3
"""Build one new runtime-K library, outside GPU timing windows."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--directory", type=Path, required=True)
args = parser.parse_args()
directory = args.directory.resolve()
llvm = Path("/root/toolchains/llvm-amdgpu-pin-op-dst-49c41889-build/bin")
saved = json.loads((ROOT / "reports/opus_9030_wide_n_20260924/build_command.json").read_text())
command = [saved[0], *saved[1:saved.index("-shared")], "-shared", "-I" + str(directory),
           "-I" + str(ROOT / "csrc/opus_gemm/include/gfx950"),
           "-I" + str(ROOT / "csrc/opus_gemm/include"), "-I" + str(ROOT / "csrc/include"),
           "-mllvm", "-verify-machineinstrs", str(directory / "launch.hip"),
           "-o", str(directory / "experiments.so")]
sources = [Path(__file__).resolve(), *sorted(directory.glob("*.cuh")),
           *sorted(directory.glob("*.hip")), *sorted(directory.glob("*.py")), directory / "variants.json",
           *sorted((ROOT / "csrc/opus_gemm/include/gfx950").glob("*.cuh")),
           ROOT / "csrc/include/opus/opus.hpp", ROOT / "csrc/include/opus/dtypes.hpp"]
manifest = dict(status="running", cpu_only=True, gpu_executed=False, command=command,
                source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
start = time.time()
with (directory / "build.log").open("w") as log:
    result = subprocess.run(command, cwd=ROOT,
        env=dict(os.environ, HIP_CLANG_PATH=str(llvm), GPU_ARCHS="gfx950", CU_NUM="256",
                 HIP_VISIBLE_DEVICES="", CUDA_VISIBLE_DEVICES="", ROCR_VISIBLE_DEVICES=""),
        stdout=log, stderr=subprocess.STDOUT)
manifest.update(status="passed" if result.returncode == 0 else "failed",
                returncode=result.returncode, elapsed_seconds=time.time() - start)
if result.returncode == 0:
    manifest["binary_sha256"] = hashlib.sha256((directory / "experiments.so").read_bytes()).hexdigest()
(directory / "build_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(dict(directory=str(directory), status=manifest["status"],
                     returncode=result.returncode, elapsed_seconds=manifest["elapsed_seconds"])), flush=True)
raise SystemExit(result.returncode)
