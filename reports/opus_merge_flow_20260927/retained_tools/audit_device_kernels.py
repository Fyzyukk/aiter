#!/usr/bin/env python3
"""Read HIP device entries and instruction bytes without loading HIP or a GPU."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess
import tempfile

LLVM = Path("/root/toolchains/llvm-amdgpu-pin-op-dst-49c41889-build/bin")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def elf_sections(data):
    head = struct.unpack_from("<16sHHIQQQIHHHHHH", data)
    if head[0][:6] != b"\x7fELF\x02\x01":
        raise ValueError("Expected little-endian ELF64")
    sections = [struct.unpack_from("<IIQQQQIIQQ", data, head[6] + i * head[11])
                for i in range(head[12])]
    names_section = sections[head[13]]
    names = data[names_section[4]:names_section[4] + names_section[5]]
    return sections, {names[s[0]:names.index(0, s[0])].decode(): s for s in sections}


def device_objects(binary):
    data = Path(binary).read_bytes()
    _, sections = elf_sections(data)
    section = sections[".hip_fatbin"]
    fat = data[section[4]:section[4] + section[5]]
    magic = b"__CLANG_OFFLOAD_BUNDLE__"
    if not fat.startswith(magic):
        raise ValueError("Expected uncompressed HIP offload bundle")
    pos = len(magic)
    count, = struct.unpack_from("<Q", fat, pos)
    pos += 8
    result = []
    for _ in range(count):
        offset, size, length = struct.unpack_from("<QQQ", fat, pos)
        pos += 24
        target = fat[pos:pos + length].decode()
        pos += length
        if target.startswith("hip-"):
            if not target.endswith("--gfx950") or offset + size > len(fat):
                raise ValueError(f"Unexpected device bundle: {target}")
            result.append((target, fat[offset:offset + size]))
    if not result:
        raise ValueError("No gfx950 device code object found")
    return result


def symbol_bytes(data, symbol_type):
    sections, _ = elf_sections(data)
    result = {}
    for section in sections:
        if section[1] != 2:
            continue
        strings = sections[section[6]]
        names = data[strings[4]:strings[4] + strings[5]]
        for offset in range(section[4], section[4] + section[5], section[9]):
            nameoff, info, _, index, address, size = struct.unpack_from("<IBBHQQ", data, offset)
            if info & 15 != symbol_type or not size or index == 0:
                continue
            name = names[nameoff:names.index(0, nameoff)].decode()
            target = sections[index]
            start = target[4] + address - target[3]
            result[name] = data[start:start + size]
    return result


def function_bytes(data):
    return symbol_bytes(data, 2)


def inspect(binary, llvm=LLVM):
    binary = Path(binary).resolve()
    result = {"binary": str(binary), "binary_sha256": sha(binary.read_bytes()),
              "cpu_only": True, "gpu_executed": False, "kernels": []}
    with tempfile.TemporaryDirectory(prefix="opus-retained-audit-") as tmp:
        for number, (target, data) in enumerate(device_objects(binary)):
            path = Path(tmp) / f"device{number}.co"
            path.write_bytes(data)
            notes = subprocess.check_output([str(llvm / "llvm-readobj"), "--notes", str(path)], text=True)
            names = re.findall(r"^    \.name:\s+(\S+)\s*$", notes, re.MULTILINE)
            symbols = re.findall(r"^    \.symbol:\s+(\S+)\s*$", notes, re.MULTILINE)
            if not names or sorted(symbols) != sorted(name + ".kd" for name in names):
                raise ValueError("Kernel metadata names/descriptors differ")
            functions = function_bytes(data)
            objects = symbol_bytes(data, 1)
            for name in names:
                body = functions[name]
                descriptor = objects[name + ".kd"]
                if len(descriptor) != 64:
                    raise ValueError(f"Unexpected kernel descriptor size: {name}")
                # The relative code-entry address changes when another geometry
                # is removed or the link layout changes. All resource/config
                # fields, including LDS, scratch, register and kernarg sizes,
                # must remain byte-identical to the measured descriptor.
                normalized_descriptor = descriptor[:16] + bytes(8) + descriptor[24:]
                result["kernels"].append({"name": name, "target": target,
                    "instruction_bytes": len(body), "instruction_sha256": sha(body),
                    "descriptor_sha256": sha(normalized_descriptor)})
    names = [kernel["name"] for kernel in result["kernels"]]
    if len(names) != len(set(names)):
        raise ValueError("Repeated kernel entry across device objects")
    result["kernel_count"] = len(names)
    return result


def audit_package(package, llvm=LLVM):
    package = Path(package).resolve()
    manifest = json.loads((package / "retained_manifest.json").read_text())
    expected_binaries = {str(Path(library["name"]) / "experiments.so") for library in manifest["libraries"]}
    actual_binaries = set()
    for path in package.rglob("*"):
        if path.is_file():
            with path.open("rb") as stream:
                magic = stream.read(4)
            if magic == b"\x7fELF" or path.suffix in {".so", ".co", ".o", ".bc"}:
                actual_binaries.add(str(path.relative_to(package)))
    if actual_binaries != expected_binaries:
        raise ValueError(f"Package contains missing/extra kernel binaries: {actual_binaries ^ expected_binaries}")
    rows = []
    for library in manifest["libraries"]:
        current = inspect(package / library["name"] / "experiments.so", llvm)
        actual = {kernel["name"]: kernel for kernel in current["kernels"]}
        expected = {kernel["name"]: kernel for kernel in library["expected_kernels"]}
        current["exact_selected_entries"] = actual.keys() == expected.keys()
        current["instruction_bytes_match_measured"] = actual.keys() == expected.keys() and all(
            actual[name]["instruction_sha256"] == expected[name]["instruction_sha256"]
            and actual[name]["instruction_bytes"] == expected[name]["instruction_bytes"]
            for name in actual)
        current["descriptor_matches_measured"] = actual.keys() == expected.keys() and all(
            actual[name]["descriptor_sha256"] == expected[name]["descriptor_sha256"] for name in actual)
        current["library"] = library["name"]
        current["ids"] = library["ids"]
        rows.append(current)
    result = {"status": "passed" if all(row["exact_selected_entries"]
                 and row["instruction_bytes_match_measured"] and row["descriptor_matches_measured"]
                 for row in rows) else "failed",
              "cpu_only": True, "gpu_executed": False, "libraries": rows,
              "actual_kernel_count": sum(row["kernel_count"] for row in rows)}
    (package / "device_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--binary", type=Path)
    group.add_argument("--package", type=Path)
    parser.add_argument("--llvm", type=Path, default=LLVM)
    args = parser.parse_args()
    result = inspect(args.binary, args.llvm) if args.binary else audit_package(args.package, args.llvm)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result.get("status", "passed") == "passed" else 1)
