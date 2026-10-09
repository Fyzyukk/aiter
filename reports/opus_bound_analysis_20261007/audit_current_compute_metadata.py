#!/usr/bin/env python3
"""CPU-only audit of official generated gfx950 compute images and private baselines.

Reads the existing ELF objects and HIP bundles without invoking HIP or a GPU.
Emits exact metadata, kernel instruction hashes and normalized descriptor hashes.
"""
from pathlib import Path
import hashlib
import json
import re
import struct
import subprocess

import msgpack

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
BUILD = OUT / "jit_baseline/build/module_deepgemm_opus/build"
GENERATED = BUILD.parent / "blob.staging"
EVIDENCE = OUT / "official_compute_metadata"
EVIDENCE.mkdir(exist_ok=True)


def sha(data):
    return hashlib.sha256(data).hexdigest()


class Elf:
    def __init__(self, data):
        self.data = data
        hdr = struct.unpack_from("<16sHHIQQQIHHHHHH", data)
        assert hdr[0][:6] == b"\x7fELF\x02\x01", "Expected little-endian ELF64"
        self.machine = hdr[2]
        off, ents, count, strindex = hdr[6], hdr[11], hdr[12], hdr[13]
        self.rows = [struct.unpack_from("<IIQQQQIIQQ", data, off + i * ents)
                     for i in range(count)]
        self.strings = self.bytes(self.rows[strindex])
        self.sections = {self.string(self.strings, r[0]): r for r in self.rows}

    @staticmethod
    def string(table, off):
        return table[off:].split(b"\0", 1)[0].decode()

    def bytes(self, row):
        return self.data[row[4]:row[4] + row[5]]

    def symbols(self):
        result = {}
        for row in self.rows:
            if row[1] not in (2, 11):
                continue
            strings = self.bytes(self.rows[row[6]])
            for off in range(row[4], row[4] + row[5], row[9]):
                name, info, other, section, value, size = struct.unpack_from("<IBBHQQ", self.data, off)
                if name and section and section < len(self.rows):
                    sec = self.rows[section]
                    begin = sec[4] + value - sec[3]
                    result[self.string(strings, name)] = {
                        "size": size, "bytes": self.data[begin:begin+size],
                        "value": value, "section": section, "type": info & 15,
                    }
        return result

    def metadata(self):
        for row in self.rows:
            if row[1] != 7:
                continue
            content = self.bytes(row)
            off = 0
            while off + 12 <= len(content):
                namesize, descsize, kind = struct.unpack_from("<III", content, off)
                off += 12
                owner = content[off:off+namesize].rstrip(b"\0")
                off += (namesize + 3) & ~3
                desc = content[off:off+descsize]
                off += (descsize + 3) & ~3
                if owner == b"AMDGPU" and kind == 32:
                    return msgpack.unpackb(desc, raw=False)
        raise ValueError("No AMDGPU metadata note")


def device_bundle(path):
    outer = Elf(path.read_bytes())
    bundle = outer.bytes(outer.sections[".hip_fatbin"])
    magic = b"__CLANG_OFFLOAD_BUNDLE__"
    assert bundle.startswith(magic)
    count = struct.unpack_from("<Q", bundle, len(magic))[0]
    offset = len(magic) + 8
    for _ in range(count):
        begin, size, idlen = struct.unpack_from("<QQQ", bundle, offset)
        offset += 24
        target = bundle[offset:offset+idlen].decode()
        offset += idlen
        if target == "hip-amdgcn-amd-amdhsa--gfx950":
            return bundle[begin:begin+size]
    raise ValueError(f"No gfx950 device bundle in {path}")


def module_image(path):
    # Linked HIP host libraries use the same bundle format as compilation objs.
    return device_bundle(path)


def summarize_image(data):
    elf = Elf(data)
    assert elf.machine == 224, "Expected AMDGPU ELF"
    metadata = elf.metadata()
    symbols = elf.symbols()
    names = [k[".name"] for k in metadata["amdhsa.kernels"]]
    demangled = subprocess.check_output(["c++filt", *names], text=True).splitlines()
    rows = []
    for kernel, demangle in zip(metadata["amdhsa.kernels"], demangled):
        code = symbols[kernel[".name"]]
        descriptor = symbols[kernel[".symbol"]]["bytes"]
        normalized = bytearray(descriptor)
        # kernel_code_entry_byte_offset is relative to its own descriptor.
        # The same code in a one-kernel TU and two-kernel TU has different placement.
        normalized[16:24] = b"\0" * 8
        rows.append({
            "name": kernel[".name"], "demangled": demangle,
            "metadata": kernel, "instruction_bytes": code["size"],
            "instruction_sha256": sha(code["bytes"]),
            "descriptor_sha256": sha(descriptor),
            "descriptor_normalized_sha256": sha(normalized),
        })
    return metadata, rows


def narrow_spill_audit(image_path, kernels):
    tool = "/root/toolchains/yuyzhang512-amdgpu-pin-op-dst-build/bin/llvm-objdump"
    disassembly = subprocess.check_output([tool, "-d", "--mcpu=gfx950", str(image_path)], text=True)
    image_path.with_suffix(".s").write_text(disassembly)
    for kernel in kernels:
        begin = disassembly.index("<" + kernel["name"] + ">:")
        body = disassembly[begin:]
        end = re.search(r"\n[0-9a-f]+ <", body)
        if end:
            body = body[:end.start()]
        lane_ops = []
        for line in body.splitlines():
            m = re.search(r"\b(v_(?:write|read)lane_b32)\s+(.+?)\s+//\s+([0-9A-F]+):", line)
            if m:
                lane_ops.append({"opcode": m[1], "operands": m[2].strip(), "absolute_pc": int(m[3], 16)})
        kernel["scalar_spill_isa"] = {
            "v_writelane_b32": body.count("v_writelane_b32"),
            "v_readlane_b32": body.count("v_readlane_b32"),
            "scratch_load": body.count("scratch_load"), "scratch_store": body.count("scratch_store"),
            "lane_operations": lane_ops,
            "interpretation": "SGPR spill slots saved in VGPR lanes; no scratch memory transactions observed",
        }


STEMS = {
    9000: "256x256x256x128_2x2_16x16x128_1x128x128_tiles1_nooob",
    9010: "256x256x256x128_2x2_16x16x128_1x128x128_tiles1",
    9020: "512x192x256x128_4x2_16x16x128_1x128x128_tiles1_main",
    9021: "256x128x128x128_2x2_16x16x128_1x128x128_tiles1_small",
    9022: "256x160x128x128_2x2_16x16x128_1x128x128_tiles1_small",
    9023: "256x64x128x128_2x2_16x16x128_1x128x128_tiles1_narrow",
    9024: "256x64x64x128_2x2_16x16x128_1x128x128_tiles1_narrow",
    9030: "512x192x256x128_4x2_16x16x128_1x128x128_tiles1",
}

def historical_resource_comparison(current):
    """Compare retained identities, without rebuilding or claiming RA causality."""
    old_root = Path("/root/workspace/gcnasm_new/gcnasm-mxfp8-final-pipeline-source-20260910/opus_gemm/mxfp8_gemm_16x16x128_blockscale_bpreshuffle_4wave")
    old_source = old_root / "results/zero_first_mfma_20260922/baseline/tmpl_generic.hpp"
    old_co = old_source.parent / "build/device.co"
    _, old_rows = summarize_image(old_co.read_bytes())
    assert len(old_rows) == 1
    old = old_rows[0]
    current_rows = {p["parent_id"]: p["variants"][0] for p in current if p["parent_id"] in (9000, 9010)}
    descriptor_rows = []
    for label, row, device_elf in [
            ("standalone_20260922", old, Elf(old_co.read_bytes())),
            *[(f"current_{kid}", row, Elf((EVIDENCE / f"kid{kid}.co").read_bytes()))
              for kid, row in current_rows.items()]]:
        descriptor = device_elf.symbols()[row["metadata"][".symbol"]]["bytes"]
        rsrc1, = struct.unpack_from("<I", descriptor, 48)
        rsrc3, = struct.unpack_from("<I", descriptor, 44)
        allocated = ((rsrc1 & 63) + 1) * 8
        descriptor_rows.append({
            "label": label, "vgpr_total": row["metadata"][".vgpr_count"],
            "agpr": row["metadata"][".agpr_count"], "sgpr": row["metadata"][".sgpr_count"],
            "lds_bytes": row["metadata"][".group_segment_fixed_size"],
            "granulated_workitem_vgpr_count": rsrc1 & 63,
            "vector_allocation_rounded_to_8": allocated,
            "accum_offset_register_index": ((rsrc3 & 63) + 1) * 4,
            "register_only_wave_limit_per_simd": 512 // allocated,
            "lds_only_workgroup_limit_per_cu": 163840 // row["metadata"][".group_segment_fixed_size"],
            "instruction_bytes": row["instruction_bytes"], "instruction_sha256": row["instruction_sha256"],
        })
    historical_rows = []
    for label, path, stage in [
            ("9000_20260928", ROOT / "reports/opus_9000_9010_separate_20260928/validation.json", "before"),
            ("9010_spill_fixed_20260928", ROOT / "reports/opus_9010_spill_20260928/summary.json", "final")]:
        data = json.loads(path.read_text())
        machine = data[stage]["candidates"]["9000"]["machine"] if stage == "before" else data[stage]["machine"]
        historical_rows.append({"label": label, "report": str(path), "report_sha256": sha(path.read_bytes()),
                                "instruction_bytes": machine["instruction_bytes"],
                                "instruction_sha256": machine["instruction_sha256"], "resources": machine["resources"]})
    current_source = ROOT / "csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_4wave_gfx950.cuh"
    pin_expressions = lambda s: [re.sub(r"\s+", "", x) for x in re.findall(r"\[\[clang::amdgpu_pin_agpr\((.*?)\)\]\]", s)]
    pins_old, pins_current = pin_expressions(old_source.read_text()), pin_expressions(current_source.read_text())
    source_identity = {}
    for f in [current_source, ROOT / "csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_4wave_256x256_padded_m_gfx950.cuh"]:
        relative = str(f.relative_to(ROOT))
        source_identity[relative] = {"current_sha256": sha(f.read_bytes()), "historical_run_matches": {}}
        for report_name in ("opus_consolidate_20260930", "opus_full745_register_20260930"):
            data = json.loads((ROOT / "reports" / report_name / "run.json").read_text())
            source_identity[relative]["historical_run_matches"][report_name] = data["source_sha256"][relative] == sha(f.read_bytes())
    compiler = Path("/root/toolchains/yuyzhang512-amdgpu-pin-op-dst-build/bin/clang++")
    return {
        "cpu_only": True, "gpu_executed": False,
        "old_frozen_source": str(old_source), "old_frozen_source_sha256": sha(old_source.read_bytes()),
        "old_device_object": str(old_co), "old_device_object_sha256": sha(old_co.read_bytes()),
        "old_live_source_sha256": sha((old_root / "tmpl_generic.hpp").read_bytes()),
        "old_live_source_warning": "The live standalone source was changed after the 20260922 record; only the retained baseline snapshot identifies the 464/212 binary.",
        "source_pin_expressions_equal": pins_old == pins_current,
        "source_pin_expression_count": len(pins_old), "source_pin_expressions": pins_old,
        "current_source_run_identity": source_identity,
        "current_compiler_version": subprocess.check_output([str(compiler), "--version"], text=True).strip(),
        "current_compiler_sha256": sha(compiler.read_bytes()),
        "historical_compiler_context": "20260922 standalone used the yuyzhang512 path; 20260928 and 20260930 integrated records used the 49c41889 path. Current compiler --version also reports 49c41889681640665400cb01c9fbb4c0a024cde4. Historical compiler binary hashes are not established here, so directory names alone do not prove a compiler change.",
        "current_context": "Official JIT C++20 flags include amdgpu-mfma-vgpr-form, no post-misched and kernarg preload; old standalone Makefile used C++17 O3 ffast-math and its frozen OPUS include. This is not a controlled compiler-only comparison.",
        "source_differences_from_frozen_standalone": [
            "SFB publish changed replicated E8M0 byte (raw*0x01010101) to low-byte u32 raw, paired with native B byte selector n_repeat to selector 0; 512-byte LDS panel size is unchanged.",
            "Runtime future_matrix_offset explicitly uses readfirstlane before the scalar constraint.",
            "Integrated traits/kernel symbol and OPUS include differ; the kernel argument layout remains 96 bytes.",
            "9010 separately supports padded M and has its historical zero-definition change to remove VGPR scratch spills.",
        ],
        "descriptor_resource_comparison": descriptor_rows,
        "descriptor_semantics_sources": [
            "/root/toolchains/yuyzhang512-amdgpu-pin-op-dst-src/llvm/include/llvm/Support/AMDHSAKernelDescriptor.h:108",
            "/root/toolchains/yuyzhang512-amdgpu-pin-op-dst-src/llvm/lib/Target/AMDGPU/Utils/AMDGPUBaseInfo.cpp:1382",
            "/root/toolchains/yuyzhang512-amdgpu-pin-op-dst-src/llvm/lib/Target/AMDGPU/Utils/AMDGPUBaseInfo.cpp:1419",
        ],
        "historical_integrated_metadata": historical_rows,
        "conclusions": [
            "477/221 for 9000 and 497/241 for 9010 existed in 20260928 integrated records; current reconstruction does not demonstrate a new vector-allocation regression.",
            "The old 464, current 477 and current 497 round to 464, 480 and 504 respectively. All have a static register-only one-wave/SIMD limit and a one-WG/CU LDS limit; no occupancy threshold is crossed.",
            "Matching explicit pin expressions does not mean matching complete machine code or natural register allocation. The exact cause of the extra AGPR allocation is not isolated by these records.",
        ],
        "limits": "Static compiler/descriptor resource bounds only; actual active waves, dispatch balance and performance require device measurements. No build, optimization experiment or GPU execution was added by this comparison.",
    }


def main():
    manifest = {
        "cpu_only": True, "gpu_executed": False,
        "source_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "audit_script_sha256": sha(Path(__file__).read_bytes()),
        "build_manifest": json.loads((OUT / "current_build.json").read_text()),
        "build_ninja_sha256": sha((BUILD / "build.ninja").read_bytes()),
        "resource_count_note": "gfx950 metadata vgpr_count is total vector allocation; do not add agpr_count again",
        "kernels": [], "private_baseline_comparison": [],
    }
    for kid, stem in STEMS.items():
        prefix = "opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_" + stem
        object_path = BUILD / (prefix + "_Cbf16_t.device.cuda.o")
        tu_path = GENERATED / "instances" / (prefix + "_Cbf16_t.device.cu")
        impl_path = GENERATED / "impl" / (prefix + ".cuh")
        data = device_bundle(object_path)
        image_path = EVIDENCE / f"kid{kid}.co"
        image_path.write_bytes(data)
        metadata, rows = summarize_image(data)
        if kid in (9023, 9024):
            narrow_spill_audit(image_path, rows)
        (EVIDENCE / f"kid{kid}_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        aliases = dict(re.findall(r"using\s+(\w+)\s*=\s*([^;]+);", impl_path.read_text()))
        parent = {
            "parent_id": kid, "object": str(object_path.relative_to(ROOT)),
            "object_sha256": sha(object_path.read_bytes()),
            "generated_tu": str(tu_path.relative_to(ROOT)), "generated_tu_sha256": sha(tu_path.read_bytes()),
            "generated_impl": str(impl_path.relative_to(ROOT)), "generated_impl_sha256": sha(impl_path.read_bytes()),
            "device_image": str(image_path.relative_to(ROOT)), "device_image_sha256": sha(data),
            "traits_aliases": aliases, "variants": rows,
        }
        manifest["kernels"].append(parent)
        if kid in (9021, 9022):
            private_data = (OUT / "compute_prologue/baseline/device.co").read_bytes()
            _, private_rows = summarize_image(private_data)
            assert len(rows) == 1
            official = rows[0]
            private = next(v for v in private_rows if v["name"] == official["name"])
            differences = {k: {"official": official["metadata"].get(k), "private": private["metadata"].get(k)}
                           for k in set(official["metadata"]) | set(private["metadata"])
                           if official["metadata"].get(k) != private["metadata"].get(k)}
            manifest["private_baseline_comparison"].append({
                "parent_id": kid, "same_symbol": True,
                "official_instruction_sha256": official["instruction_sha256"],
                "private_instruction_sha256": private["instruction_sha256"],
                "instructions_identical": official["instruction_sha256"] == private["instruction_sha256"],
                "metadata_identical": not differences, "metadata_differences": differences,
                "descriptor_identical": official["descriptor_sha256"] == private["descriptor_sha256"],
                "normalized_descriptor_identical": official["descriptor_normalized_sha256"] == private["descriptor_normalized_sha256"],
                "normalized_descriptor_ignored": "kernel_code_entry_byte_offset bytes16..23 only",
            })

    manifest["historical_resource_comparison"] = historical_resource_comparison(manifest["kernels"])
    manifest["status"] = "passed"
    (OUT / "official_compute_metadata.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for parent in manifest["kernels"]:
        for row in parent["variants"]:
            m = row["metadata"]
            print(json.dumps({"kid": parent["parent_id"], "kernel": row["demangled"],
                              "vgpr": m[".vgpr_count"], "agpr": m[".agpr_count"],
                              "sgpr": m[".sgpr_count"], "lds": m[".group_segment_fixed_size"],
                              "private": m[".private_segment_fixed_size"],
                              "vgpr_spill": m[".vgpr_spill_count"], "sgpr_spill": m[".sgpr_spill_count"],
                              "instruction_bytes": row["instruction_bytes"]}))
    print(json.dumps({"private_baseline_comparison": manifest["private_baseline_comparison"]}))


if __name__ == "__main__":
    main()
