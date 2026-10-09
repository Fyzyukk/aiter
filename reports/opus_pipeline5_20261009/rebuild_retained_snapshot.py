#!/usr/bin/env python3
"""Compile the hashed historical retained TUs offline with frozen dependencies.

This check emits objects in a new directory. It does not link shared libraries,
execute the historical builder/auditor, or load any binary or GPU runtime.
"""

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PACKAGE_REL = Path("csrc/opus_gemm/mxfp8_bpreshuffle_retained")
HISTORICAL_LLVM = Path("/root/toolchains/llvm-amdgpu-pin-op-dst-49c41889-build/bin")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def file_record(path):
    return {"exists": path.is_file(), "sha256": sha(path) if path.is_file() else None}


def source_hashes(package, manifest):
    actual = {name: sha(package / name) for name in manifest["source_sha256"]}
    changed = [name for name, digest in manifest["source_sha256"].items()
               if actual[name] != digest]
    if changed:
        raise ValueError(f"Historical package source changed: {changed}")
    return actual


def historical_artifacts(package, manifest):
    names = ["retained_manifest.json", "build_manifest.json", "device_audit.json"]
    names += [f"{row['name']}/{name}" for row in manifest["libraries"]
              for name in ("build_manifest.json", "build.log", "experiments.so")]
    return {name: file_record(package / name) for name in names}


def compile_command(template, library, root, llvm, dependencies, obj, resource_dir):
    """Retain compile flags; replace link action/output with a HIP object action."""
    command = [arg.replace("@LIBRARY@", str(library)).replace("@ROOT@", str(root))
               .replace("@LLVM@", str(llvm)) for arg in template]
    command[0] = str(llvm / "clang++")
    if "-x" not in command:
        command[1:1] = ["-x", "hip"]
    live_include = "-I" + str(root / "csrc/opus_gemm/include/gfx950")
    if command.count(live_include) != 1:
        raise ValueError("Expected one live gfx950 include in historical template")
    command.insert(command.index(live_include), "-I" + str(dependencies))
    compiled = []
    skip = False
    for arg in command:
        if skip:
            skip = False
            continue
        if arg == "-o":
            skip = True
        elif arg == "-shared" or arg.startswith(("-L", "-l")):
            continue
        else:
            compiled.append(arg)
    if "-DOPUS_ENABLE_RUNTIME_QUERY=0" not in compiled:
        compiled.append("-DOPUS_ENABLE_RUNTIME_QUERY=0")
    if not any(arg.startswith("--rocm-path=") for arg in compiled):
        compiled.append("--rocm-path=/opt/rocm")
    if resource_dir:
        compiled.append("-resource-dir=" + str(resource_dir))
    return compiled + ["-c", "-o", str(obj)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--llvm", type=Path, default=HISTORICAL_LLVM,
                        help="compiler bin directory; changing it checks source compatibility only")
    parser.add_argument("--resource-dir", type=Path,
                        help="optional clang resource directory for the selected compiler")
    parser.add_argument("--dependency-include", type=Path,
                        help="defaults to the saved before/csrc/opus_gemm/include/gfx950 headers")
    parser.add_argument("--output-dir", type=Path, default=HERE / "retained_compile",
                        help="new directory; existing directories are never overwritten")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    root, llvm, output = args.root.resolve(), args.llvm.resolve(), args.output_dir.resolve()
    package = root / PACKAGE_REL
    before = root / "reports/opus_pipeline5_20261009/before"
    dependencies = (args.dependency_include.resolve() if args.dependency_include else
                    before / "csrc/opus_gemm/include/gfx950")
    resource_dir = args.resource_dir.resolve() if args.resource_dir else None
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if not (llvm / "clang++").is_file():
        parser.error(f"Compiler is absent: {llvm / 'clang++'}; select --llvm explicitly")
    if not (dependencies / "opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_4wave_gfx950.cuh").is_file():
        parser.error(f"Historical dependency header is absent: {dependencies}")
    if output.is_relative_to(package) or output.is_relative_to(before):
        parser.error("Output must be outside the historical package and before snapshot")
    if output.exists():
        parser.error(f"Refusing to overwrite output directory: {output}")

    manifest = json.loads((package / "retained_manifest.json").read_text())
    original_hashes = source_hashes(package, manifest)
    protected = historical_artifacts(package, manifest)
    for name in ("retained_manifest.json", "build_manifest.json", "device_audit.json"):
        if sha(package / name) != sha(before / PACKAGE_REL / name):
            raise ValueError(f"Historical record differs from before snapshot: {name}")
    audit = json.loads((package / "device_audit.json").read_text())
    if audit["status"] != "passed" or audit["actual_kernel_count"] != 11:
        raise ValueError("Expected the frozen passed audit of eleven historical kernels")
    expected_binaries = {row["library"]: row["binary_sha256"] for row in audit["libraries"]}
    for name, digest in expected_binaries.items():
        binary = package / name / "experiments.so"
        if binary.is_file() and sha(binary) != digest:
            raise ValueError(f"Historical binary differs from frozen audit: {binary}")

    output.mkdir(parents=True)
    sources, objects, logs = output / "sources", output / "objects", output / "logs"
    objects.mkdir()
    logs.mkdir()
    for name in manifest["source_sha256"]:
        target = sources / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(package / name, target)
    copied_hashes = source_hashes(sources, manifest)
    dependency_hashes = {str(path.relative_to(dependencies.parent)): sha(path)
                         for path in sorted(dependencies.rglob("*")) if path.is_file()}
    dependency_hashes["opus_gemm_utils.cuh"] = sha(dependencies.parent / "opus_gemm_utils.cuh")
    environment = dict(os.environ, HIP_CLANG_PATH=str(llvm), GPU_ARCHS="gfx950", CU_NUM="256",
                       HIP_VISIBLE_DEVICES="", CUDA_VISIBLE_DEVICES="", ROCR_VISIBLE_DEVICES="")
    commands = []
    for row in manifest["libraries"]:
        command = compile_command(row["build_command_template"], sources / row["name"], root,
                                  llvm, dependencies, objects / f"{row['name']}.o", resource_dir)
        commands.append(dict(library=row["name"], ids=row["ids"], argv=command))
    (output / "compile_commands.json").write_text(json.dumps(commands, indent=2) + "\n")

    def compile_one(row):
        log = logs / f"{row['library']}.log"
        start = time.monotonic()
        with log.open("w") as stream:
            process = subprocess.run(row["argv"], cwd=root, env=environment,
                                     stdout=stream, stderr=subprocess.STDOUT)
        obj = objects / f"{row['library']}.o"
        result = dict(row, returncode=process.returncode, seconds=time.monotonic() - start,
                      log=str(log), log_sha256=sha(log), object=str(obj),
                      object_sha256=sha(obj) if obj.is_file() else None)
        print(json.dumps({"library": row["library"], "returncode": process.returncode}), flush=True)
        return result

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        builds = list(pool.map(compile_one, commands))
    hashes_after = source_hashes(package, manifest)
    protected_after = historical_artifacts(package, manifest)
    unchanged = protected_after == protected and hashes_after == original_hashes
    passed = unchanged and all(row["returncode"] == 0 for row in builds)
    result = dict(status="passed" if passed else "failed", compile_only=True, compile_count=len(builds),
                  source_hash_count=len(original_hashes), source_sha256=original_hashes,
                  copied_source_sha256=copied_hashes, dependency_include=str(dependencies),
                  dependency_sha256=dependency_hashes, compiler=str(llvm / "clang++"),
                  compiler_sha256=sha(llvm / "clang++"), resource_dir=str(resource_dir) if resource_dir else None,
                  historical_compiler_selected=llvm == HISTORICAL_LLVM.resolve(),
                  wrapper_sha256=sha(__file__), frozen_artifacts_unchanged=unchanged,
                  historical_artifacts_before=protected, historical_artifacts_after=protected_after,
                  historical_audit_kernel_count=audit["actual_kernel_count"],
                  historical_binary_sha256_from_audit=expected_binaries,
                  present_historical_binaries=sum((package / name / "experiments.so").is_file()
                                                  for name in expected_binaries),
                  instruction_bytes_compared=False, gpu_queries=0, gpu_runtime_initializations=0,
                  library_loads=0, shared_libraries_linked=0, kernel_launches=0, builds=builds)
    (output / "verification_receipt.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "compile_count": len(builds),
                      "source_hash_count": len(original_hashes), "historical_artifacts_unchanged": unchanged}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
