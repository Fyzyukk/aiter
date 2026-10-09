#!/usr/bin/env python3
"""Offline Clang23 build of frozen runtime control and private short-K kernels.

Never imports torch/aiter, loads the libraries, or launches HIP operations.
Existing build outputs are refused; all argv, logs and hashes stay local.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CLANG = Path("/opt/rocm-llvm23-46fcb339/bin/clang++")
ENV = dict(os.environ, ROCR_VISIBLE_DEVICES="", HIP_VISIBLE_DEVICES="", CUDA_VISIBLE_DEVICES="")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    source_manifest = json.loads((HERE / "source_manifest.json").read_text())
    common = [str(CLANG), "-x", "hip", "--rocm-path=/opt/rocm", "--hip-path=/opt/rocm",
              "--offload-arch=gfx950", "-std=c++20", "-O3", "-fPIC", "-fvisibility=hidden",
              "-D__HIP_PLATFORM_AMD__=1", "-DOPUS_ENABLE_RUNTIME_QUERY=0",
              "-fgpu-flush-denormals-to-zero", "-fno-offload-uniform-block", "-fno-gpu-rdc",
              "-mllvm", "--amdgpu-kernarg-preload-count=32",
              "-mllvm", "--amdgpu-mfma-vgpr-form", "-mllvm", "--lsr-drop-solution=1",
              "-mllvm", "-amdgpu-early-inline-all=true", "-mllvm", "-amdgpu-function-calls=false",
              "-mllvm", "-enable-post-misched=0", "-mllvm", "-verify-machineinstrs",
              "-I" + str(HERE), "-I" + str(HERE / "frozen"),
              "-I" + str(HERE / "frozen/gemm_include"),
              "-I" + str(HERE / "frozen/gemm_include/gfx950")]
    commands = []
    for side in ["baseline", "candidate"]:
        folder = HERE / "build" / side
        object_path = folder / "launch.o"
        library_path = folder / "experiments.so"
        compile_argv = common + (["-DSHORTK9022_CANDIDATE=1"] if side == "candidate" else [])
        compile_argv += ["-c", str(HERE / "launch.hip"), "-o", str(object_path)]
        # The fatbinary registration is emitted by the HIP compile; the host link never runs it.
        link_argv = [str(CLANG), "-shared", str(object_path), "-L/opt/rocm/lib", "-lamdhip64",
                     "-Wl,--build-id=sha1", "-o", str(library_path)]
        commands.append({"side": side, "compile_argv": compile_argv, "link_argv": link_argv,
                         "object": str(object_path), "library": str(library_path)})
    if args.dry_run:
        print(json.dumps({"cpu_only": True, "status": "dry_run", "builds": commands}, indent=2))
        return
    if (HERE / "build").exists() or (HERE / "build_receipt.json").exists():
        raise SystemExit("Refusing to overwrite an offline build or its receipt")
    manifest_before = sha(HERE / "source_manifest.json")
    production_before = {row["original"]: sha(ROOT / row["original"]) for row in source_manifest["frozen_files"]}
    assert all(production_before[row["original"]] == row["sha256"] for row in source_manifest["frozen_files"])
    formal = source_manifest["formal_control"]
    assert sha(Path(formal["path"])) == formal["sha256"]
    receipt = {"status": "building", "cpu_only": True, "numerical_validation": "not_run_gpu_stopped",
               "performance_validation": "not_run_gpu_stopped", "registered": False,
               "compiler": {"path": str(CLANG), "sha256": sha(CLANG),
                            "version": subprocess.check_output([str(CLANG), "--version"], env=ENV, text=True),
                            "resource_dir": subprocess.check_output([str(CLANG), "-print-resource-dir"], env=ENV, text=True).strip()},
               "source_manifest_sha256": manifest_before,
               "sources": {str(p.relative_to(HERE)): sha(p) for p in sorted(HERE.rglob("*"))
                           if p.is_file() and p.suffix in {".hip", ".h", ".cuh", ".hpp"}},
               "builds": commands, "formal_control": formal}
    for build in commands:
        folder = Path(build["object"]).parent
        folder.mkdir(parents=True)
        for phase in ["compile", "link"]:
            started = time.monotonic()
            completed = subprocess.run(build[phase + "_argv"], cwd=HERE, env=ENV, text=True,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            log = folder / (phase + ".log")
            log.write_text(completed.stdout)
            build[phase + "_returncode"] = completed.returncode
            build[phase + "_seconds"] = time.monotonic() - started
            build[phase + "_log_sha256"] = sha(log)
            if completed.returncode:
                receipt["status"] = "failed_" + phase
                (HERE / "build_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
                print(completed.stdout)
                raise SystemExit(completed.returncode)
        build["object_sha256"] = sha(Path(build["object"]))
        build["library_sha256"] = sha(Path(build["library"]))
        print(json.dumps({"side": build["side"], "offline_build": "passed",
                          "object_sha256": build["object_sha256"]}), flush=True)
    unchanged = all(sha(ROOT / p) == value for p, value in production_before.items())
    assert unchanged and sha(HERE / "source_manifest.json") == manifest_before
    assert sha(Path(formal["path"])) == formal["sha256"]
    receipt.update(status="offline_build_passed_unvalidated_numerics", production_sources_unchanged=unchanged,
                   formal_control_unchanged=True)
    (HERE / "build_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
