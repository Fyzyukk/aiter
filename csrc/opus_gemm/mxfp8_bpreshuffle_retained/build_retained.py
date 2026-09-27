#!/usr/bin/env python3
"""Build only selected retained libraries, then audit exact device entries/bytes."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from audit_device_kernels import LLVM, audit_package, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--llvm", type=Path, default=LLVM)
    args = parser.parse_args()
    package = args.package.resolve()
    root = args.root.resolve() if args.root else next(
        path for path in package.parents if (path / "csrc/include/opus/opus.hpp").is_file())
    manifest = json.loads((package / "retained_manifest.json").read_text())
    for path, digest in manifest["source_sha256"].items():
        if sha((package / path).read_bytes()) != digest:
            raise ValueError(f"Prepared source changed: {path}")
    environment = dict(os.environ, HIP_CLANG_PATH=str(args.llvm), GPU_ARCHS="gfx950", CU_NUM="256",
                       HIP_VISIBLE_DEVICES="", CUDA_VISIBLE_DEVICES="", ROCR_VISIBLE_DEVICES="")
    result = {"status": "running", "cpu_only": True, "gpu_executed": False, "libraries": []}
    for library in manifest["libraries"]:
        directory = package / library["name"]
        command = [arg.replace("@LIBRARY@", str(directory)).replace("@ROOT@", str(root))
                   .replace("@LLVM@", str(args.llvm)) for arg in library["build_command_template"]]
        start = time.time()
        with (directory / "build.log").open("w") as log:
            process = subprocess.run(command, cwd=root, env=environment, stdout=log, stderr=subprocess.STDOUT)
        row = {"library": library["name"], "ids": library["ids"], "command": command,
               "returncode": process.returncode, "elapsed_seconds": time.time() - start}
        result["libraries"].append(row)
        if process.returncode:
            result["status"] = "failed"
            (package / "build_manifest.json").write_text(json.dumps(result, indent=2) + "\n")
            raise SystemExit(process.returncode)
        row["binary_sha256"] = sha((directory / "experiments.so").read_bytes())
        local_build = dict(row, status="passed", cpu_only=True, gpu_executed=False,
                           source_sha256={str(directory / path.name): sha(path.read_bytes())
                                          for path in directory.iterdir()
                                          if path.suffix in {".cuh", ".hip"}
                                          or path.name == "variants.json"})
        (directory / "build_manifest.json").write_text(json.dumps(local_build, indent=2) + "\n")
        print(json.dumps(row), flush=True)
    audit = audit_package(package, args.llvm)
    result["status"] = audit["status"]
    result["device_audit_sha256"] = sha((package / "device_audit.json").read_bytes())
    (package / "build_manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "kernel_count": audit["actual_kernel_count"]}))
    raise SystemExit(0 if result["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
