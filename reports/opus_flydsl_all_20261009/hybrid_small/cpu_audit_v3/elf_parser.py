#!/usr/bin/env python3
"""Audit existing SFA-packed HIP binaries on CPU; never build or load HIP.

All generated files stay beside this script. The Oct 7 private and formal
selected objects are read-only references. Instruction hashes use the complete
ELF FUNC symbol bytes; descriptor normalization clears only bytes 16..23.
"""
import collections
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import subprocess

import msgpack

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OLD = ROOT / "reports/opus_bound_analysis_20261007"
LLVM = Path("/root/toolchains/yuyzhang512-amdgpu-pin-op-dst-build/bin")
TARGET = "hip-amdgcn-amd-amdhsa--gfx950"
ENV = dict(os.environ, ROCR_VISIBLE_DEVICES="", HIP_VISIBLE_DEVICES="", CUDA_VISIBLE_DEVICES="")
COMMANDS = []


def digest(data):
    return hashlib.sha256(data).hexdigest()


def sha(path):
    return digest(Path(path).read_bytes())


def run(args):
    command = [str(v) for v in args]
    COMMANDS.append(command)
    return subprocess.check_output(command, cwd=ROOT, env=ENV, text=True)


class Elf:
    def __init__(self, data):
        self.data = data
        h = struct.unpack_from("<16sHHIQQQIHHHHHH", data)
        assert h[0][:6] == b"\x7fELF\x02\x01", "Expected little-endian ELF64"
        self.machine = h[2]
        self.rows = [struct.unpack_from("<IIQQQQIIQQ", data, h[6] + i * h[11]) for i in range(h[12])]
        strings = self.bytes(self.rows[h[13]])
        self.sections = {self.string(strings, r[0]): r for r in self.rows}

    @staticmethod
    def string(table, offset):
        return table[offset:].split(b"\0", 1)[0].decode()

    def bytes(self, row):
        return self.data[row[4]:row[4] + row[5]]

    def symbols(self):
        result = {}
        for row in self.rows:
            if row[1] not in (2, 11):
                continue
            strings = self.bytes(self.rows[row[6]])
            for offset in range(row[4], row[4] + row[5], row[9]):
                name, info, _, section, value, size = struct.unpack_from("<IBBHQQ", self.data, offset)
                if name and 0 < section < len(self.rows):
                    sec = self.rows[section]
                    begin = sec[4] + value - sec[3]
                    result[self.string(strings, name)] = {"size": size, "value": value,
                        "type": info & 15, "bytes": self.data[begin:begin + size]}
        return result

    def metadata(self):
        for row in self.rows:
            if row[1] != 7:
                continue
            content, offset = self.bytes(row), 0
            while offset + 12 <= len(content):
                namesize, descsize, kind = struct.unpack_from("<III", content, offset)
                offset += 12
                owner = content[offset:offset + namesize].rstrip(b"\0")
                offset += (namesize + 3) & ~3
                desc = content[offset:offset + descsize]
                offset += (descsize + 3) & ~3
                if owner == b"AMDGPU" and kind == 32:
                    return msgpack.unpackb(desc, raw=False)
        raise ValueError("No AMDGPU metadata note")


def bundled_image(path):
    elf = Elf(path.read_bytes())
    bundle = elf.bytes(elf.sections[".hip_fatbin"])
    magic = b"__CLANG_OFFLOAD_BUNDLE__"
    assert bundle.startswith(magic)
    count, = struct.unpack_from("<Q", bundle, len(magic))
    offset = len(magic) + 8
    for _ in range(count):
        begin, size, length = struct.unpack_from("<QQQ", bundle, offset)
        offset += 24
        target = bundle[offset:offset + length].decode()
        offset += length
        if target == TARGET:
            return bundle[begin:begin + size]
    raise ValueError(f"No gfx950 bundle in {path}")


def audit_image(input_path, label):
    folder = HERE / "device_audit_artifacts" / label
    folder.mkdir(parents=True, exist_ok=True)
    fat, device, inspection = folder / "fatbin.bin", folder / "device.co", folder / "inspection_copy.o"
    input_before = sha(input_path)
    run([LLVM / "llvm-objcopy", "--dump-section=.hip_fatbin=" + str(fat), input_path, inspection])
    inspection.unlink()
    targets = run([LLVM / "clang-offload-bundler", "-type=o", "-list", "-input=" + str(fat)]).splitlines()
    assert TARGET in targets, targets
    run([LLVM / "clang-offload-bundler", "-type=o", "-unbundle", "-targets=" + TARGET,
         "-input=" + str(fat), "-output=" + str(device)])
    data = device.read_bytes()
    assert data == bundled_image(input_path), "LLVM and independent bundle parser disagree"
    notes = run([LLVM / "llvm-readelf", "--notes", device])
    isa = run([LLVM / "llvm-objdump", "-d", "--mcpu=gfx950", device])
    symbols_text = run([LLVM / "llvm-readelf", "--symbols", device])
    (folder / "metadata.txt").write_text(notes)
    (folder / "isa.txt").write_text(isa)
    (folder / "symbols.txt").write_text(symbols_text)
    elf = Elf(data)
    assert elf.machine == 224
    metadata, symbols = elf.metadata(), elf.symbols()
    (folder / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    names = [r[".name"] for r in metadata["amdhsa.kernels"]]
    demangled = run(["c++filt", *names]).splitlines()
    blocks = {}
    matches = list(re.finditer(r"^([0-9a-fA-F]+) <([^>]+)>:\n", isa, re.M))
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(isa)
        blocks[match.group(2)] = isa[match.end():end]
    kernels = []
    keys = ["agpr_count", "vgpr_count", "sgpr_count", "group_segment_fixed_size",
            "private_segment_fixed_size", "vgpr_spill_count", "sgpr_spill_count"]
    for kernel, name in zip(metadata["amdhsa.kernels"], demangled):
        kid = 9021 if "4wave_128x128_traits_gfx950" in name else 9022 if "4wave_160x128_traits_gfx950" in name else None
        assert kid is not None, name
        code = symbols[kernel[".name"]]
        assert code["type"] == 2 and len(code["bytes"]) == code["size"] > 0
        descriptor = symbols[kernel[".symbol"]]["bytes"]
        assert len(descriptor) == 64
        normalized = bytearray(descriptor)
        normalized[16:24] = b"\0" * 8
        body = blocks[kernel[".name"]]
        (folder / f"kid{kid}.s").write_text(body)
        counts = collections.Counter()
        for line in body.splitlines():
            match = re.match(r"\s+([a-z][a-z0-9_]+)\b.*//\s+([0-9A-Fa-f]+):", line)
            if match and code["value"] <= int(match[2], 16) < code["value"] + code["size"]:
                counts[match[1]] += 1
        ds_reads = {k: v for k, v in counts.items() if k.startswith("ds_read")}
        ds_writes = {k: v for k, v in counts.items() if k.startswith("ds_write")}
        kernels.append({"kid": kid, "name": kernel[".name"], "demangled": name, "metadata": kernel,
            "resources": {key: kernel["." + key] for key in keys},
            "instruction_bytes": code["size"], "instruction_sha256": digest(code["bytes"]),
            "descriptor_sha256": digest(descriptor), "descriptor_normalized_sha256": digest(normalized),
            "instruction_counts": dict(sorted(counts.items())),
            "static_summary": {"all_instructions": sum(counts.values()),
                "scaled_mfma": sum(v for k, v in counts.items() if k.startswith("v_mfma_scale")),
                "ds_read_total": sum(ds_reads.values()), "ds_write_total": sum(ds_writes.values()),
                "ds_reads": dict(sorted(ds_reads.items())), "ds_writes": dict(sorted(ds_writes.items())),
                "buffer_load_ubyte": counts["buffer_load_ubyte"],
                "buffer_load_dwordx4": counts["buffer_load_dwordx4"],
                "v_readlane_b32": counts["v_readlane_b32"], "v_writelane_b32": counts["v_writelane_b32"],
                "scratch_load": sum(v for k, v in counts.items() if k.startswith("scratch_load")),
                "scratch_store": sum(v for k, v in counts.items() if k.startswith("scratch_store"))}})
    assert input_before == sha(input_path), "Input object changed during inspection"
    return {"label": label, "input": str(input_path), "input_sha256": input_before,
        "input_unchanged": True, "target": TARGET, "bundle_targets": targets,
        "device_sha256": sha(device), "device": str(device), "metadata_sha256": sha(folder / "metadata.json"),
        "isa_sha256": sha(folder / "isa.txt"), "kernels": sorted(kernels, key=lambda x: x["kid"])}


def identity(a, b):
    result = {"symbol_equal": a["name"] == b["name"],
        "instruction_equal": a["instruction_sha256"] == b["instruction_sha256"],
        "instruction_bytes_equal": a["instruction_bytes"] == b["instruction_bytes"],
        "metadata_equal": a["metadata"] == b["metadata"],
        "descriptor_normalized_equal": a["descriptor_normalized_sha256"] == b["descriptor_normalized_sha256"]}
    result["matches"] = all(result.values())
    return result


def main():
    report = {"status": "running", "cpu_only": True, "gpu_executed": False, "build_executed": False,
        "visibility": {k: ENV[k] for k in ["ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"]},
        "audit_script_sha256": sha(Path(__file__)), "tools": {}}
    for name in ["llvm-objcopy", "clang-offload-bundler", "llvm-readelf", "llvm-objdump"]:
        report["tools"][name] = {"path": str(LLVM / name), "version": run([LLVM / name, "--version"]).strip()}
    inputs = [p for side in ["baseline", "candidate"] for p in (HERE / side).rglob("*")
              if p.is_file() and p.suffix in [".so", ".hip", ".cuh"]]
    before = {str(p.relative_to(HERE)): sha(p) for p in inputs}
    manifest = json.loads((HERE / "build_manifest.json").read_text())
    assert manifest["status"] == "passed" and manifest["cpu_only"]
    report["build_manifest_sha256"] = sha(HERE / "build_manifest.json")
    for build in manifest["builds"]:
        assert sha(HERE / build["side"] / "experiments.so") == build["library_sha256"]
    selected_path = OLD / "formal_selected/identity_audit.json"
    selected = json.loads(selected_path.read_text())
    report["selected_identity_record_sha256"] = sha(selected_path)
    libraries = [audit_image(HERE / side / "experiments.so", side) for side in ["baseline", "candidate"]]
    assert all({r["kid"] for r in v["kernels"]} == {9021, 9022} for v in libraries)
    references = [audit_image(OLD / "compute_prologue" / side / "experiments.so", "oct7_prologue_" + side)
                  for side in ["baseline", "candidate"]]
    for kid in [9021, 9022]:
        parent = next(p for p in selected["parents"] if p["parent_id"] == kid)
        path = Path(parent["candidate_object"])
        assert sha(path) == parent["candidate_object_sha256"], "Frozen selected object hash mismatch"
        ref = audit_image(path, f"oct7_selected_{kid}")
        assert len(ref["kernels"]) == len(parent["variants"]) == 1
        assert identity(ref["kernels"][0], parent["variants"][0]["candidate"])["matches"]
        ref["matches_frozen_selected_record"] = True
        references.append(ref)
    report["libraries"], report["references"] = libraries, references
    lookup = {(v["label"], r["kid"]): r for v in libraries + references for r in v["kernels"]}
    comparisons = []
    for left, right, kid in [("baseline", "oct7_prologue_candidate", 9021),
        ("baseline", "oct7_selected_9021", 9021), ("baseline", "oct7_prologue_baseline", 9022),
        ("candidate", "oct7_prologue_baseline", 9022), ("baseline", "oct7_selected_9022", 9022),
        ("candidate", "oct7_selected_9022", 9022), ("baseline", "candidate", 9022)]:
        comparisons.append({"left": left, "right": right, "kid": kid,
                            "identity": identity(lookup[left, kid], lookup[right, kid])})
    report["required_identity_comparisons"] = comparisons
    changes = []
    for kid in [9021, 9022]:
        baseline, candidate = lookup["baseline", kid], lookup["candidate", kid]
        opcodes = set(baseline["instruction_counts"]) | set(candidate["instruction_counts"])
        changes.append({"kid": kid, "identity": identity(baseline, candidate),
            "resources_baseline": baseline["resources"], "resources_candidate": candidate["resources"],
            "resource_deltas": {k: candidate["resources"][k] - baseline["resources"][k] for k in baseline["resources"]},
            "instruction_bytes_baseline": baseline["instruction_bytes"],
            "instruction_bytes_candidate": candidate["instruction_bytes"],
            "opcode_deltas": {k: candidate["instruction_counts"].get(k, 0) - baseline["instruction_counts"].get(k, 0)
                              for k in sorted(opcodes) if candidate["instruction_counts"].get(k, 0) != baseline["instruction_counts"].get(k, 0)},
            "static_summary_baseline": baseline["static_summary"], "static_summary_candidate": candidate["static_summary"]})
    report["changes"] = changes
    after = {str(p.relative_to(HERE)): sha(p) for p in inputs}
    report["source_and_binary_inputs_sha256"] = before
    report["source_and_binary_inputs_unchanged"] = before == after
    report["commands"] = COMMANDS
    report["summary"] = {"baseline_matches_oct7_selected": all(c["identity"]["matches"] for c in comparisons if c["left"] == "baseline"),
        "candidate_9022_unchanged": all(c["identity"]["matches"] for c in comparisons if c["kid"] == 9022),
        "candidate_9021_instructions_changed": not changes[0]["identity"]["instruction_equal"],
        "candidate_9021_kernarg_and_lds_unchanged": all(lookup["baseline", 9021]["metadata"][k] == lookup["candidate", 9021]["metadata"][k]
                                                    for k in [".kernarg_segment_size", ".group_segment_fixed_size"]),
        "all_scratch_and_spill_zero": all(r["resources"][k] == 0 for v in libraries for r in v["kernels"]
                                         for k in ["private_segment_fixed_size", "vgpr_spill_count", "sgpr_spill_count"])}
    report["limits"] = ["Instruction counts are static ELF FUNC-symbol counts, not dynamic executions or bytes transferred.",
        "Static resources and unchanged input hashes do not establish GPU correctness or performance.",
        "Only descriptor bytes16..23 (kernel_code_entry_byte_offset) are normalized; all instruction bytes and metadata are exact."]
    passed = all(c["identity"]["matches"] for c in comparisons) and before == after and all(report["summary"].values())
    report["status"] = "passed" if passed else "failed_requires_review"
    (HERE / "device_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "summary": report["summary"], "changes": changes}, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
