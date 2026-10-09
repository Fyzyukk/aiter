#!/usr/bin/env python3
"""Compare corresponding compiled device entries by reading ELF bytes only."""
import hashlib
import importlib.util
import json
from pathlib import Path
import struct

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASELINE = ROOT / "reports/opus_runtime_splitk_20261009/full_build"
PARSER = ROOT / "reports/opus_flydsl_all_20261009/hybrid_small/cpu_audit_v4/elf_parser.py"
spec = importlib.util.spec_from_file_location("offline_elf_parser", PARSER)
parser = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parser)

def device_image(path):
    elf = parser.Elf(path.read_bytes())
    data = elf.bytes(elf.sections[".hip_fatbin"])
    magic = b"__CLANG_OFFLOAD_BUNDLE__"
    begin = data.index(magic)
    count, = struct.unpack_from("<Q", data, begin + len(magic))
    cursor = begin + len(magic) + 8
    found = []
    for _ in range(count):
        start, size, length = struct.unpack_from("<QQQ", data, cursor)
        cursor += 24
        target = data[cursor:cursor + length].decode()
        cursor += length
        if "amdgcn" in target and "gfx950" in target:
            found.append(data[begin + start:begin + start + size])
    assert len(found) == 1
    return parser.Elf(found[0])

def normalized_metadata(kernel):
    metadata = dict(kernel)
    metadata.pop(".name", None)
    metadata.pop(".symbol", None)
    return metadata

def main():
    old = {row["kid"]: row for row in json.loads((BASELINE / "build_receipt.json").read_text())["builds"]}
    builds = json.loads((HERE / "build_receipt.json").read_text())["builds"]
    resource_names = ("sgpr_count", "vgpr_count", "agpr_count", "group_segment_fixed_size",
                      "private_segment_fixed_size", "vgpr_spill_count", "sgpr_spill_count")
    rows = []
    for row in builds:
        kid = row["kid"]
        prior = device_image(ROOT / old[kid]["object"])
        device = device_image(ROOT / row["object"])
        prior_kernels, kernels = prior.metadata()["amdhsa.kernels"], device.metadata()["amdhsa.kernels"]
        assert len(prior_kernels) == len(kernels), (kid, len(prior_kernels), len(kernels))
        prior_funcs, funcs = prior.symbols(), device.symbols()
        instances = []
        # Generation preserves the explicit specialization order per object.
        # Both complete names are recorded so every pairing remains reviewable.
        for old_kernel, kernel in zip(prior_kernels, kernels):
            old_func, func = prior_funcs[old_kernel[".name"]], funcs[kernel[".name"]]
            old_resources = {key: old_kernel.get("." + key, 0) for key in resource_names}
            resources = {key: kernel.get("." + key, 0) for key in resource_names}
            instances.append(dict(
                prior_kernel=old_kernel[".name"], kernel=kernel[".name"],
                instruction_byte_identical=old_func["bytes"] == func["bytes"],
                prior_instruction_bytes=old_func["size"], instruction_bytes=func["size"],
                prior_instruction_sha256=hashlib.sha256(old_func["bytes"]).hexdigest(),
                instruction_sha256=hashlib.sha256(func["bytes"]).hexdigest(),
                metadata_ignoring_private_symbol_names_identical=normalized_metadata(old_kernel) == normalized_metadata(kernel),
                resources=resources, prior_resources=old_resources,
                resource_delta={key: [old_resources[key], resources[key]] for key in resource_names if old_resources[key] != resources[key]},
                kernarg_size=kernel.get(".kernarg_segment_size")))
        rows.append(dict(kid=kid, kernel_count=len(instances), instances=instances))
    instances = [item for row in rows for item in row["instances"]]
    result = dict(status="comparison_completed", configuration_count=len(rows),
                  emitted_instance_count=len(instances),
                  instruction_byte_identical_instances=sum(item["instruction_byte_identical"] for item in instances),
                  metadata_identical_instances_ignoring_private_symbol_names=sum(item["metadata_ignoring_private_symbol_names_identical"] for item in instances),
                  resource_changed_instances=sum(bool(item["resource_delta"]) for item in instances),
                  scope="Corresponding emitted metadata entries in preserved instantiation order; private kernel and symbol names excluded from metadata equality. Numerical and performance behavior remain untested.",
                  all_objects=rows)
    (HERE / "instruction_metadata_comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "all_objects"}, indent=2))

if __name__ == "__main__":
    main()
