#!/usr/bin/env python3
"""Reproduce this experiment's existing CPU-only device identity audit.

Reuse the SFA audit's ELF extraction, metadata, ISA and exact-symbol hashing
helpers, with HERE redirected to this experiment. The audited Oct8 SFA baseline
is a read-only identity reference. No compile or GPU/HIP loading is performed.
Running the script recreates this directory's audit artifacts/device_audit.json;
the source trees, libraries and build manifest are checked for unchanged hashes.
"""
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHARED = HERE.parent / "sfa_packed"


def main():
    spec = importlib.util.spec_from_file_location("shared_sfa_device_audit",
                                                SHARED / "audit_device.py")
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    audit.HERE = HERE
    audit.COMMANDS.clear()
    manifest_path = HERE / "build_manifest.json"
    inputs = [manifest_path] + [p for side in ["baseline", "candidate"]
                              for p in (HERE / side).rglob("*")
                              if p.is_file() and p.suffix in [".so", ".hip", ".cuh"]]
    before = {str(p.relative_to(HERE)): audit.sha(p) for p in inputs}
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "passed" and manifest["cpu_only"]
    for build in manifest["builds"]:
        assert audit.sha(HERE / build["side"] / "experiments.so") == build["library_sha256"]
    reference_path = SHARED / "device_audit.json"
    reference_before = audit.sha(reference_path)
    reference_report = json.loads(reference_path.read_text())
    assert reference_report["status"] == "passed" and reference_report["cpu_only"]
    reference = next(v for v in reference_report["libraries"] if v["label"] == "baseline")
    assert audit.sha(reference["input"]) == reference["input_sha256"]
    assert audit.sha(reference["device"]) == reference["device_sha256"]
    refs = {k["kid"]: k for k in reference["kernels"]}
    libraries = [audit.audit_image(HERE / side / "experiments.so", side)
                 for side in ["baseline", "candidate"]]
    assert all({k["kid"] for k in v["kernels"]} == {9021, 9022} for v in libraries)
    lookup = {(v["label"], k["kid"]): k for v in libraries for k in v["kernels"]}
    comparisons = [{"label": label, "kid": kid,
                    "identity": audit.identity(lookup[label, kid], refs[kid])}
                   for label, kid in [("baseline", 9021), ("baseline", 9022), ("candidate", 9022)]]
    after = {str(p.relative_to(HERE)): audit.sha(p) for p in inputs}
    assert audit.sha(reference_path) == reference_before
    passed = before == after and all(v["input_unchanged"] for v in libraries) and all(
        c["identity"]["matches"] for c in comparisons)
    # Preserve the original experiment audit's schema for direct reproduction.
    report = {"status": "passed" if passed else "failed_requires_review",
              "cpu_only": True, "libraries": libraries,
              "required_identity_comparisons": comparisons,
              "sources_unchanged": before == after}
    (HERE / "device_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "libraries"}, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
