#!/usr/bin/env python3
"""Read full HIP objects and linked ELF bytes on CPU; do not load libraries."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import struct
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
GENERATED = HERE.parent / "codegen"
ENV = dict(os.environ, ROCR_VISIBLE_DEVICES="", HIP_VISIBLE_DEVICES="", CUDA_VISIBLE_DEVICES="")
LLVM = Path("/opt/rocm-llvm23-46fcb339/bin")
PARSER = ROOT / "reports/opus_flydsl_all_20261009/hybrid_small/cpu_audit_v4/elf_parser.py"
spec = importlib.util.spec_from_file_location("offline_elf_parser", PARSER)
parser = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parser)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def command(argv):
    proc = subprocess.run([str(x) for x in argv], cwd=ROOT, env=ENV, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return proc.stdout


def symbols(path, flag):
    text = command([LLVM / "llvm-nm", flag, path])
    return {line.split()[-1] for line in text.splitlines() if line.strip()}, text


def images(data):
    magic = b"__CLANG_OFFLOAD_BUNDLE__"
    offset = 0
    while True:
        begin = data.find(magic, offset)
        if begin < 0:
            return
        count, = struct.unpack_from("<Q", data, begin + len(magic))
        cursor = begin + len(magic) + 8
        records = []
        for _ in range(count):
            image_start, size, length = struct.unpack_from("<QQQ", data, cursor)
            cursor += 24
            target = data[cursor:cursor + length].decode()
            cursor += length
            records.append((target, image_start, size))
        for target, image_start, size in records:
            if "amdgcn" in target and "gfx950" in target:
                yield target, data[begin + image_start:begin + image_start + size]
        offset = begin + max(start + size for _, start, size in records)


def run():
    output = HERE / "final_receipt.json"
    if output.exists():
        raise SystemExit("Refusing existing inspection receipt")
    build = json.loads((HERE / "build_receipt.json").read_text())
    assert build["status"] == "full105_generated_hip_object_compilation_passed"
    host_refs, host_text = symbols(GENERATED / "all_instances_host_gfx950.o", "--undefined-only")
    stub_refs = {name for name in host_refs if "__device_stub__" in name and "bpreshuffle" in name}
    assert len(stub_refs) == 123
    (HERE / "host_undefined_symbols.txt").write_text(host_text)
    definitions = set()
    object_stubs = {}
    for row in build["builds"]:
        path = ROOT / row["object"]
        assert sha(path) == row["object_sha256"]
        defined, _ = symbols(path, "--defined-only")
        definitions |= defined
        object_stubs[row["kid"]] = sorted(name for name in defined if "__device_stub__" in name)
    assert stub_refs <= definitions, sorted(stub_refs - definitions)
    linked = HERE / "all105_registered.so"
    linked_undefined, linked_undefined_text = symbols(linked, "--undefined-only")
    linked_defined, linked_defined_text = symbols(linked, "--defined-only")
    assert stub_refs <= linked_defined
    assert not {name for name in linked_undefined if "__device_stub__" in name}
    (HERE / "linked_undefined_symbols.txt").write_text(linked_undefined_text)
    (HERE / "linked_defined_symbols.txt").write_text(linked_defined_text)
    dynamics = command([LLVM / "llvm-readelf", "-d", linked])
    (HERE / "linked_dynamic.txt").write_text(dynamics)

    aggregate = json.loads((HERE.parent / "header_promotion/final_receipt.json").read_text())
    expected = {row["id"]: row for side in aggregate["builds"] for row in side["kernels"]}
    metadata = {row["kid"]: row for row in json.loads((GENERATED / "all105_metadata.json").read_text())}
    new_resources = []
    all_image_kernel_counts = {}
    linked_device_symbols = set()
    linked_elf = parser.Elf(linked.read_bytes())
    for _, image in images(linked_elf.bytes(linked_elf.sections[".hip_fatbin"])):
        elf = parser.Elf(image)
        linked_device_symbols.update(kernel[".name"] for kernel in elf.metadata()["amdhsa.kernels"])
    for row in build["builds"]:
        elf = parser.Elf((ROOT / row["object"]).read_bytes())
        bundles = list(images(elf.bytes(elf.sections[".hip_fatbin"])))
        assert len(bundles) == 1, row["kid"]
        target, image = bundles[0]
        device = parser.Elf(image)
        kernels, funcs = device.metadata()["amdhsa.kernels"], device.symbols()
        all_image_kernel_counts[row["kid"]] = len(kernels)
        if not metadata[row["kid"]]["new_variant"]:
            continue
        reference = expected[row["kid"]]
        main = next(k for k in kernels if k[".name"] == reference["metadata"][".name"])
        name = main[".name"]
        match = re.match(r"_Z(\d+)(.*)", name)
        stub = "_Z" + str(int(match[1]) + 15) + "__device_stub__" + match[2]
        assert stub in stub_refs and stub in object_stubs[row["kid"]]
        assert name in linked_device_symbols
        resources = {key: main.get("." + key, 0) for key in reference["resources"]}
        assert all(resources[key] == 0 for key in ("private_segment_fixed_size", "vgpr_spill_count", "sgpr_spill_count"))
        function = funcs[name]
        assert function["type"] == 2 and function["size"] > 0
        instruction_sha = hashlib.sha256(function["bytes"]).hexdigest()
        resource_differences = {key: dict(per_id=resources[key], aggregate=reference["resources"][key])
                                for key in resources if resources[key] != reference["resources"][key]}
        new_resources.append(dict(kid=row["kid"], compiler=row["side"], kernel=name,
                                  host_stub=stub, resources=resources, instruction_bytes=function["size"],
                                  instruction_sha256=instruction_sha, bundled_device_sha256=hashlib.sha256(image).hexdigest(),
                                  device_kernel_count_including_reducer=len(kernels),
                                  aggregate_resource_differences=resource_differences,
                                  aggregate_function_bytes_equal=(function["size"] == reference["instruction_bytes"] and
                                                                  instruction_sha == reference["instruction_sha256"])))
    assert len(new_resources) == 64
    result = dict(status="all105_generated_full_hip_compile_and_no_undefined_link_verified_gpu_pending",
                  registered_ids=105, public_tuning_candidates=92, new_variants=64,
                  compile_count=105, compiler_counts={side: sum(row["side"] == side for row in build["builds"])
                                                      for side in ("main23", "pin24")},
                  all105_object_hashes_match=True, host_kernel_stub_reference_count=len(stub_refs),
                  all123_host_kernel_references_defined_in_objects_and_linked_library=True,
                  linked_kernel_stub_undefined_count=0, all64_new_host_device_symbols_exact_match=True,
                  all64_new_device_functions_present_in_linked_fatbin=True,
                  aggregate_reference_comparison="Actual per-ID resources and complete function hashes are recorded. Different compilation translation units can change allocation and ISA; identity with the aggregate build is not asserted.",
                  aggregate_resource_difference_ids=[row["kid"] for row in new_resources if row["aggregate_resource_differences"]],
                  aggregate_function_difference_ids=[row["kid"] for row in new_resources if not row["aggregate_function_bytes_equal"]],
                  all64_new_scratch_and_spills_zero=True,
                  all105_device_image_kernel_counts=all_image_kernel_counts,
                  total_device_kernel_instances_including_duplicated_shared_specializations=sum(all_image_kernel_counts.values()),
                  new64_resources=new_resources, object_defined_stub_symbols=object_stubs,
                  linked_library="all105_registered.so", linked_library_sha256=sha(linked),
                  link_argv_file="link_argv.json", link_log_file="link.log", link_returncode=0,
                  link_no_undefined=True,
                  runtime_dependencies="Linked ELF records libamdhip64 as the normal launch-runtime dependency. It was not loaded or executed.",
                  host_stub_explanation="The 105 IDs include saved compatibility entries, dispatch specializations, and shared reducers. Their complete generated host TU references 123 distinct HIP launch stubs; every one is now supplied by the full per-ID HIP objects and the linked ELF.",
                  superseded_evidence_limit="The prior aggregate header objects were device-only with empty host bundles; full generated per-ID objects and link close that prior linkage gap.",
                  failed_attempts=[dict(kind="inspection_assertion", status="superseded", log="inspection_attempts.json",
                                        reason="The initial inspection assumed aggregate and per-ID TU resource/ISA identity. Pin24 allocation and ISA differ while scratch and spills remain zero; final receipt records actual per-ID resources.")],
                  gpu_queries=0, gpu_module_imports=0, libraries_loaded=False,
                  kernels_launched=0, executables_run=False,
                  numerical_validation="not_run_gpu_stopped", performance_validation="not_run_gpu_stopped",
                  build_receipt_sha256=sha(HERE / "build_receipt.json"),
                  aggregate_receipt_sha256=sha(HERE.parent / "header_promotion/final_receipt.json"),
                  codegen_receipt_sha256=sha(GENERATED / "codegen_receipt.json"),
                  inspection_script_sha256=sha(__file__), elf_parser_sha256=sha(PARSER),
                  files={str(p.relative_to(HERE)): sha(p) for p in sorted(HERE.rglob("*"))
                         if p.is_file() and p != output})
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(status=result["status"], host_stubs=123, new_resources=64,
                          receipt_sha256=sha(output), linked_library_sha256=sha(linked))))


if __name__ == "__main__":
    run()
