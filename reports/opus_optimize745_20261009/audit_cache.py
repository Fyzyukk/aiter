#!/usr/bin/env python3
"""Audit exact runtime module images on CPU without loading HIP libraries."""
import hashlib
import importlib.util
import json
from pathlib import Path
import struct

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def device_image(parser, data):
    host = parser.Elf(data)
    bundle = host.bytes(host.sections[".hip_fatbin"])
    magic = b"__CLANG_OFFLOAD_BUNDLE__"
    assert bundle.startswith(magic)
    count, = struct.unpack_from("<Q", bundle, len(magic))
    offset = len(magic) + 8
    targets, matches = [], []
    for _ in range(count):
        begin, size, length = struct.unpack_from("<QQQ", bundle, offset)
        offset += 24
        target = bundle[offset:offset + length].decode()
        offset += length
        targets.append(target)
        if target.endswith("--gfx950"):
            image = bundle[begin:begin + size]
            assert len(image) == size
            matches.append(image)
    assert len(matches) == 1, targets
    return matches[0], targets


def main():
    path = ROOT / "reports/opus_resume_20261008/sfa_packed/audit_device.py"
    spec = importlib.util.spec_from_file_location("cache745_elf", path)
    parser = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parser)
    receipt_path = HERE / "cache_build.json"
    receipt_bytes = receipt_path.read_bytes()
    receipt = json.loads(receipt_bytes)
    plan = json.loads((HERE / "plan.json").read_text())
    assert receipt["completed"] == receipt["total"] == len(plan["configs"]) == 204
    assert {r["identity"] for r in receipt["rows"]} == set(plan["configs"])
    rows = []
    resource_keys = ("agpr_count", "vgpr_count", "sgpr_count",
                     "group_segment_fixed_size", "private_segment_fixed_size",
                     "vgpr_spill_count", "sgpr_spill_count")
    for build in receipt["rows"]:
        assert build["returncode"] == 0
        artifact = Path(build["path"])
        data = artifact.read_bytes()
        assert digest(data) == build["sha256"], artifact
        image, targets = device_image(parser, data)
        elf = parser.Elf(image)
        assert elf.machine == 224
        symbols = elf.symbols()
        kernels = []
        for metadata in elf.metadata()["amdhsa.kernels"]:
            code = symbols[metadata[".name"]]
            descriptor = symbols[metadata[".symbol"]]["bytes"]
            assert code["type"] == 2 and len(code["bytes"]) == code["size"] > 0
            assert len(descriptor) == 64
            kernels.append({"name": metadata[".name"],
                            "resources": {k: metadata.get("."+k, 0) for k in resource_keys},
                            "isa_bytes": code["size"], "isa_sha256": digest(code["bytes"]),
                            "descriptor_sha256": digest(descriptor)})
        assert kernels
        row = {"identity": build["identity"], "pipeline": build["pipeline"],
               "legacy_kid": plan["configs"][build["identity"]]["legacy_kid"],
               "path": str(artifact), "module_sha256": digest(data),
               "device_sha256": digest(image), "bundle_targets": targets, "kernels": kernels}
        rows.append(row)
    spills = [{"identity": r["identity"], "pipeline": r["pipeline"],
               "legacy_kid": r["legacy_kid"], "name": k["name"], "resources": k["resources"]}
              for r in rows for k in r["kernels"] if any(k["resources"][key] for key in
                  ("private_segment_fixed_size", "vgpr_spill_count", "sgpr_spill_count"))]
    assert receipt_path.read_bytes() == receipt_bytes
    report = {"configs": len(rows), "kernels": sum(len(r["kernels"]) for r in rows),
              "spilled_kernels": spills, "rows": rows, "gpu_operations": 0,
              "config_library_loads": 0, "cache_receipt_sha256": digest(receipt_bytes),
              "plan_sha256": digest((HERE / "plan.json").read_bytes()),
              "script_sha256": digest(Path(__file__).read_bytes()),
              "numerical_validation": "pending_idle_gpu",
              "performance_validation": "pending_idle_gpu"}
    destination = HERE / "cache_resources.json"
    assert not destination.exists(), "Preserve prior audit evidence"
    destination.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}))


if __name__ == "__main__":
    main()
