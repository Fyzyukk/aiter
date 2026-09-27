#!/usr/bin/env python3
"""Compile the renamed Traits kernels and compare device code with their sources."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time


REPORT = Path(__file__).resolve().parent
ROOT = REPORT.parents[1]
HEADERS = ROOT / "csrc/opus_gemm/include/gfx950"
AUDIT_TOOLS = ROOT / "reports/opus_merge_flow_20260927/retained_tools"
sys.path.insert(0, str(AUDIT_TOOLS))
from audit_device_kernels import LLVM, inspect, sha


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def verify_protected():
    snapshot = json.loads((REPORT / "protected_sha256.json").read_text())
    changed = [
        name for name, digest in snapshot.items()
        if not (ROOT / name).is_file() or sha((ROOT / name).read_bytes()) != digest
    ]
    if changed:
        raise ValueError(f"Protected source or reference artifact changed: {changed}")
    return len(snapshot)


def entry(audit, kernel, template_value):
    matches = [
        row for row in audit["kernels"]
        if kernel in row["name"]
        and (template_value is None or f"ILi{template_value}E" in row["name"])
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one {kernel}<{template_value}>: {matches}")
    return matches[0]


def build_family(spec, environment):
    reference = ROOT / spec["reference_directory"]
    directory = REPORT / spec["family"]
    reference_build = json.loads((reference / "build_manifest.json").read_text())
    command = [
        arg.replace(str(reference), str(directory))
        for arg in reference_build["command"]
    ]
    own_sources = [
        directory / "launch.hip", directory / "variants.json",
        HEADERS / spec["pipeline"],
        HEADERS / spec["pipeline"].replace("_pipeline_", "_traits_"),
    ]
    source_hashes = {str(p): sha(p.read_bytes()) for p in own_sources}
    result = {
        "library": spec["family"], "ids": spec["ids"], "status": "running",
        "cpu_only": True, "gpu_executed": False, "command": command,
        "toolchain": str(LLVM), "source_sha256": source_hashes,
    }
    started = time.monotonic()
    with (directory / "build.log").open("w") as log:
        process = subprocess.run(
            command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT
        )
    result.update(
        returncode=process.returncode,
        elapsed_seconds=time.monotonic() - started,
    )
    if process.returncode:
        result["status"] = "failed"
        write_json(directory / "build_manifest.json", result)
        raise RuntimeError(f"Compilation failed; see {directory / 'build.log'}")

    before = inspect(reference / "experiments.so")
    after = inspect(directory / "experiments.so")
    comparisons = []
    for instance in spec["instances"]:
        old = entry(before, spec["source_kernel"], instance["template_value"])
        new = entry(after, spec["kernel"], instance["template_value"])
        comparisons.append({
            "id": instance["id"], "old_symbol": old["name"], "new_symbol": new["name"],
            "instruction_bytes": new["instruction_bytes"],
            "instruction_sha256": new["instruction_sha256"],
            "instructions_identical": (
                old["instruction_bytes"] == new["instruction_bytes"]
                and old["instruction_sha256"] == new["instruction_sha256"]
            ),
            "descriptor_identical_except_code_address": (
                old["descriptor_sha256"] == new["descriptor_sha256"]
            ),
        })
    audit = {
        "cpu_only": True, "gpu_executed": False,
        "reference": before, "current": after, "comparisons": comparisons,
        "exact_kernel_count": (
            before["kernel_count"] == after["kernel_count"] == len(spec["ids"])
        ),
    }
    audit["status"] = "passed" if audit["exact_kernel_count"] and all(
        row["instructions_identical"] and row["descriptor_identical_except_code_address"]
        for row in comparisons
    ) else "failed"
    write_json(directory / "device_audit.json", audit)
    result.update(
        status=audit["status"], binary_sha256=after["binary_sha256"],
        actual_device_kernel_count=after["kernel_count"],
        device_audit_sha256=sha((directory / "device_audit.json").read_bytes()),
    )
    write_json(directory / "build_manifest.json", result)
    if audit["status"] != "passed":
        raise RuntimeError(f"Device code changed; see {directory / 'device_audit.json'}")
    print(f"{spec['family']}: {len(comparisons)} kernels; identical instructions and descriptors",
          flush=True)
    return result


def main():
    result = {"status": "running", "cpu_only": True, "gpu_executed": False, "libraries": []}
    try:
        result["protected_files"] = verify_protected()
        environment = dict(
            os.environ, HIP_CLANG_PATH=str(LLVM), GPU_ARCHS="gfx950", CU_NUM="256",
            HIP_VISIBLE_DEVICES="", CUDA_VISIBLE_DEVICES="", ROCR_VISIBLE_DEVICES="",
        )
        for spec in json.loads((REPORT / "sources.json").read_text()):
            result["libraries"].append(build_family(spec, environment))
        result["protected_files"] = verify_protected()
        result["actual_device_kernel_count"] = sum(
            row["actual_device_kernel_count"] for row in result["libraries"]
        )
        result["status"] = "passed"
    except Exception as error:
        result.update(status="failed", error=str(error))
        raise
    finally:
        write_json(REPORT / "build_manifest.json", result)
    print(f"Passed: {result['actual_device_kernel_count']} kernels; "
          f"{result['protected_files']} protected files unchanged.", flush=True)


if __name__ == "__main__":
    main()
