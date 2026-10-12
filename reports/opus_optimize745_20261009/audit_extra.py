#!/usr/bin/env python3
"""Read extra device images and record resources; never loads HIP libraries."""
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    path = ROOT / "reports/opus_resume_20261008/sfa_packed/audit_device.py"
    spec = importlib.util.spec_from_file_location("extra745_elf", path)
    parser = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parser)
    parser.TARGET = "hipv4-amdgcn-amd-amdhsa--gfx950"
    receipt = json.loads((HERE / "extra_device_build/receipt.json").read_text())
    assert receipt["completed"] == receipt["total"] == 133
    rows = []
    for build in receipt["rows"]:
        assert build["returncode"] == 0
        obj = Path(build["object"])
        assert hashlib.sha256(obj.read_bytes()).hexdigest() == build["object_sha256"]
        image = parser.bundled_image(obj)
        elf = parser.Elf(image)
        symbols = elf.symbols()
        kernels = []
        for row in elf.metadata()["amdhsa.kernels"]:
            code = symbols[row[".name"]]
            assert code["type"] == 2 and len(code["bytes"]) == code["size"] > 0
            keys = ("agpr_count", "vgpr_count", "sgpr_count", "group_segment_fixed_size",
                    "private_segment_fixed_size", "vgpr_spill_count", "sgpr_spill_count")
            kernels.append({"name": row[".name"], "resources": {k: row.get("."+k) for k in keys},
                            "isa_bytes": code["size"],
                            "isa_sha256": hashlib.sha256(code["bytes"]).hexdigest()})
        rows.append({"identity": build["identity"], "pipeline": build["pipeline"],
                     "kernels": kernels, "device_sha256": hashlib.sha256(image).hexdigest()})
    spilled = [row["identity"] for row in rows if any(
        k["resources"].get("private_segment_fixed_size", 0)
        or k["resources"].get("vgpr_spill_count", 0)
        or k["resources"].get("sgpr_spill_count", 0) for k in row["kernels"])]
    report = {"configs": len(rows), "kernels": sum(len(row["kernels"]) for row in rows),
              "spilled_configs": spilled, "rows": rows, "gpu_operations": 0,
              "numerical_validation": "pending_idle_gpu", "performance_validation": "pending_idle_gpu"}
    destination = HERE / "extra_resources.json"
    assert not destination.exists()
    destination.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}))


if __name__ == "__main__":
    main()
