#!/usr/bin/env python3
"""Inspect compiled ELF resources and run matrix, scale, chunk-output and guard checks on CPU only."""
import argparse
import difflib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
LLVM = Path("/opt/rocm-llvm23-46fcb339/bin")
ENV = dict(os.environ, ROCR_VISIBLE_DEVICES="", HIP_VISIBLE_DEVICES="", CUDA_VISIBLE_DEVICES="")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    arguments = argparse.ArgumentParser(description=__doc__)
    arguments.add_argument("--output-name", default="cpu_audit_final", help="Fresh local audit directory name")
    arguments.add_argument("--receipt-name", default="build_receipt_v2.json")
    parsed = arguments.parse_args()
    assert parsed.output_name and Path(parsed.output_name).name == parsed.output_name
    output = HERE / parsed.output_name
    if output.exists():
        raise SystemExit("Refusing to overwrite CPU audit")
    output.mkdir()
    # These helpers only parse existing ELF bytes; importing the standalone audit
    # imports msgpack and standard libraries, with no torch/aiter/HIP module load.
    parser_path = ROOT / "reports/opus_resume_20261008/sfa_packed/audit_device.py"
    spec = importlib.util.spec_from_file_location("read_only_elf_parser", parser_path)
    parser = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parser)
    # Clang23's local offline build uses the current hipv4 bundle spelling.
    parser.TARGET = "hipv4-amdgcn-amd-amdhsa--gfx950"
    receipt = json.loads((HERE / parsed.receipt_name).read_text())
    manifest = json.loads((HERE / "source_manifest.json").read_text())
    assert receipt["status"] == "offline_build_passed_unvalidated_numerics"
    for path in ["launch.hip", "contract.h", "traits.cuh", "panel16/pipeline.cuh", "directb/pipeline.cuh"]:
        assert sha(HERE / path) == receipt["sources"][path], path
    records = []
    for build in receipt["builds"]:
        library = Path(build["library"])
        assert sha(library) == build["library_sha256"]
        image = parser.bundled_image(library)
        device = output / (build["side"] + ".co")
        device.write_bytes(image)
        elf = parser.Elf(image)
        metadata, symbols = elf.metadata(), elf.symbols()
        (output / (build["side"] + "_metadata.json")).write_text(json.dumps(metadata, indent=2) + "\n")
        isa_argv = [str(LLVM / "llvm-objdump"), "-d", "--mcpu=gfx950", str(device)]
        isa = subprocess.check_output(isa_argv, cwd=HERE, env=ENV, text=True)
        (output / (build["side"] + "_isa.txt")).write_text(isa)
        kernels = []
        for row in metadata["amdhsa.kernels"]:
            code = symbols[row[".name"]]
            assert code["type"] == 2 and len(code["bytes"]) == code["size"] > 0
            name = subprocess.check_output(["c++filt", row[".name"]], env=ENV, text=True).strip()
            fields = ["agpr_count", "vgpr_count", "sgpr_count", "group_segment_fixed_size",
                      "private_segment_fixed_size", "vgpr_spill_count", "sgpr_spill_count"]
            kernels.append({"name": name, "symbol": row[".name"],
                            "resources": {key: row.get("." + key) for key in fields},
                            "isa_bytes": code["size"], "isa_sha256": hashlib.sha256(code["bytes"]).hexdigest()})
        expected = 1
        assert len(kernels) == expected
        assert all(k["resources"]["private_segment_fixed_size"] == 0 for k in kernels)
        assert all(k["resources"]["agpr_count"] == 0 and k["resources"]["vgpr_spill_count"] == 0
                   and k["resources"]["sgpr_spill_count"] == 0 for k in kernels)
        assert all(k["resources"]["group_segment_fixed_size"] == {"baseline": 143360, "panel16": 121408, "directb": 79168}[build["side"]]
                   for k in kernels)
        records.append({"side": build["side"], "device_sha256": sha(device), "kernels": kernels,
                        "isa_argv": isa_argv})

    # Exercise the actual frozen address-layout helpers as host-only functions.
    # Enable the layout and MMA adaptor section's host attribute in a separate audit
    # include. Expressions are byte-identical; the compiled HIP input stays frozen.
    audit_opus = output / "host_include/opus"
    audit_opus.mkdir(parents=True)
    opus_source = (HERE / "frozen/opus/opus.hpp").read_text()
    begin_host = opus_source.index("template<typename FDim, typename Target, index_t I0")
    end_host = opus_source.index("#undef OPUS_KP_", begin_host)
    host_source = opus_source[:begin_host] + opus_source[begin_host:end_host].replace("OPUS_D ", "OPUS_H_D ") + opus_source[end_host:]
    (audit_opus / "opus.hpp").write_text(host_source)
    (output / "host_attributes.diff").write_text("".join(difflib.unified_diff(
        opus_source.splitlines(keepends=True), host_source.splitlines(keepends=True),
        fromfile="frozen/opus/opus.hpp", tofile="cpu_audit_final/host_include/opus/opus.hpp")))
    for name in ["dtypes.hpp", "hip_minimal.hpp"]:
        (audit_opus / name).write_bytes((HERE / "frozen/opus" / name).read_bytes())
    common = (HERE / "frozen/gemm_include/gfx950/opus_gemm_mxscale_bpreshuffle_layout_gfx950.cuh").read_text()
    begin = common.index("template<class T>\n__device__ inline auto make_layout_ga_scale")
    end = common.index("template<class T>\n__device__ inline constexpr auto make_layout_gsfa_scale", begin)
    matrix_helpers = common[begin:end].replace("__device__", "__host__")
    wide = (HERE / "frozen/gemm_include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_8wave_192x256_gfx950.cuh").read_text()
    begin = wide.index("template<class T>\n__device__ inline constexpr auto make_layout_ra_scale")
    end = wide.index("} // namespace opus_gemm_8wave_192x256_layout", begin)
    helpers = wide[begin:end].replace("__device__", "__host__")
    # Compile the candidate's actual direct-B address expressions in the host audit.
    candidate = (HERE / "directb/pipeline.cuh").read_text()
    address_begin = candidate.index("        const int group = n_repeat * T::T_N + wave_id_n;")
    address_end = candidate.index("        static_for<T::B_CHUNKS_PER_FRAGMENT>", address_begin)
    direct_address = candidate[address_begin:address_end].replace("kargs.stride_b", "stride_b")
    chunk_address = "base + chunk * T::WARP_SIZE * T::VEC_B"
    assert candidate.count(chunk_address) == 1
    helpers += "template<class T>\n__host__ inline int direct_b_address(int tile_k, int n_repeat, int wave_id_n, int lane_id, int chunk, int stride_b) {\n" + direct_address + "    return " + chunk_address + ";\n}\n"
    synchronization_fragments = [
        "s_waitcnt_vmcnt(0_I);\n                v_b = v_b_next;",
        "s_waitcnt_lgkmcnt(0_I);\n        __builtin_amdgcn_s_barrier();\n        read_scales(tile_k + 1",
        "s_waitcnt_lgkmcnt(0_I);\n    __builtin_amdgcn_s_barrier();\n    static_for<T::C_CHUNKS>",
        "s_waitcnt_lgkmcnt(0_I);\n        __builtin_amdgcn_s_barrier();\n    });"]
    assert all(fragment in candidate for fragment in synchronization_fragments)
    header = output / "frozen_layout_helpers.h"
    header.write_text('#pragma once\n#include <opus/hip_minimal.hpp>\n#include <opus/opus.hpp>\nusing opus::operator""_I;\nnamespace checked_matrix {\n' + matrix_helpers + "}\nnamespace checked_layout {\n" + helpers + "}\n")
    binary = output / "layout_guard_check"
    host_object = output / "layout_guard_check.o"
    argv = [str(LLVM / "clang++"), "-x", "hip", "--offload-host-only", "--rocm-path=/opt/rocm",
            "--hip-path=/opt/rocm", "-std=c++20", "-O2", "-D__HIPCC_RTC__=1",
            "-I" + str(output), "-I" + str(HERE), "-I" + str(output / "host_include"),
            "-I" + str(HERE / "frozen"),
            "-I" + str(HERE / "frozen/gemm_include/gfx950"),
            "-c", str(HERE / "layout_guard_check.cpp"), "-o", str(host_object)]
    compiled = subprocess.run(argv, cwd=HERE, env=ENV, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (output / "layout_compile.log").write_text(compiled.stdout)
    if compiled.returncode:
        raise RuntimeError(compiled.stdout)
    link_argv = [str(LLVM / "clang++"), str(host_object), "-o", str(binary)]
    linked = subprocess.run(link_argv, cwd=HERE, env=ENV, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (output / "layout_link.log").write_text(linked.stdout)
    if linked.returncode:
        raise RuntimeError(linked.stdout)
    dynamic = subprocess.check_output([str(LLVM / "llvm-readelf"), "-d", str(binary)], cwd=HERE, env=ENV, text=True)
    (output / "layout_dynamic.txt").write_text(dynamic)
    assert "amdhip" not in dynamic and "hsa-runtime" not in dynamic
    checked = subprocess.check_output([str(binary)], cwd=HERE, env=ENV, text=True).strip()
    production_unchanged = all(sha(ROOT / row["original"]) == row["sha256"] for row in manifest["frozen_files"])
    assert production_unchanged and sha(Path(manifest["formal_control"]["path"])) == manifest["formal_control"]["sha256"]
    report = {"status": "cpu_audit_passed_unvalidated_numerics", "cpu_only": True,
              "gpu_operations": 0, "numerical_validation": "not_run_gpu_stopped",
              "performance_validation": "not_run_gpu_stopped", "libraries": records,
              "layout_guard_check": {"compile_argv": argv, "link_argv": link_argv,
                                     "host_object_sha256": sha(host_object), "binary_sha256": sha(binary),
                                     "no_hip_or_hsa_runtime_dependency": True,
                                     "source_sha256": sha(HERE / "layout_guard_check.cpp"),
                                     "helper_source_sha256": sha(header), "stdout": checked,
                                     "host_attribute_adapter_sha256": sha(audit_opus / "opus.hpp"),
                                     "host_attribute_diff_sha256": sha(output / "host_attributes.diff")},
              "production_sources_unchanged": production_unchanged, "formal_control_unchanged": True,
              "compiled_source_hashes_match": True,
              "candidate_synchronization_fragments_checked": len(synchronization_fragments),
              "build_receipt": {"path": parsed.receipt_name, "sha256": sha(HERE / parsed.receipt_name)},
              "elf_parser_reference": {"path": str(parser_path), "sha256": sha(parser_path)}}
    (HERE / "cpu_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
