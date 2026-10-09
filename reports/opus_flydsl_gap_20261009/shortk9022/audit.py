#!/usr/bin/env python3
"""Inspect compiled ELF resources and run scale-layout/guard checks on CPU only."""
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
    receipt = json.loads((HERE / "build_receipt.json").read_text())
    manifest = json.loads((HERE / "source_manifest.json").read_text())
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
        expected = 1 if build["side"] == "baseline" else 2
        assert len(kernels) == expected
        assert all(k["resources"]["private_segment_fixed_size"] == 0 for k in kernels)
        assert all(k["resources"]["group_segment_fixed_size"] == (81184 if expected == 1 else 77320)
                   for k in kernels)
        records.append({"side": build["side"], "device_sha256": sha(device), "kernels": kernels,
                        "isa_argv": isa_argv})

    # Exercise the actual frozen address-layout helpers as host-only functions.
    # Enable only the pure layout section's host attribute in a separate audit
    # include. Expressions are byte-identical; the compiled HIP input stays frozen.
    audit_opus = output / "host_include/opus"
    audit_opus.mkdir(parents=True)
    opus_source = (HERE / "frozen/opus/opus.hpp").read_text()
    begin_host = opus_source.index("template<typename FDim, typename Target, index_t I0")
    end_host = opus_source.index("#define OPUS_KP_", begin_host)
    host_source = opus_source[:begin_host] + opus_source[begin_host:end_host].replace("OPUS_D ", "OPUS_H_D ") + opus_source[end_host:]
    (audit_opus / "opus.hpp").write_text(host_source)
    (output / "host_attributes.diff").write_text("".join(difflib.unified_diff(
        opus_source.splitlines(keepends=True), host_source.splitlines(keepends=True),
        fromfile="frozen/opus/opus.hpp", tofile="cpu_audit_final/host_include/opus/opus.hpp")))
    for name in ["dtypes.hpp", "hip_minimal.hpp"]:
        (audit_opus / name).write_bytes((HERE / "frozen/opus" / name).read_bytes())
    small = (HERE / "frozen/gemm_include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_4wave_128x128_gfx950.cuh").read_text()
    begin = small.index("template<class T, int Pass>\n")
    end = small.index("} // namespace opus_gemm_4wave_128x128_layout")
    helpers = small[begin:end].replace("__device__", "__host__")
    wide = (HERE / "frozen/gemm_include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_8wave_192x256_gfx950.cuh").read_text()
    begin = wide.index("template<class T>\n__device__ inline constexpr auto make_layout_rsfa_scale")
    end = wide.index("template<class T>\n__device__ inline constexpr auto make_layout_gsfb_scale", begin)
    helpers += wide[begin:end].replace("__device__", "__host__")
    header = output / "frozen_layout_helpers.h"
    header.write_text('#pragma once\n#include <opus/hip_minimal.hpp>\n#include <opus/opus.hpp>\nusing opus::operator""_I;\nnamespace checked_layout {\n' + helpers + "}\n")
    binary = output / "layout_guard_check"
    argv = [str(LLVM / "clang++"), "-x", "hip", "--offload-host-only", "--rocm-path=/opt/rocm",
            "--hip-path=/opt/rocm", "-std=c++20", "-O2", "-D__HIPCC_RTC__=1",
            "-I" + str(output), "-I" + str(HERE), "-I" + str(output / "host_include"),
            "-I" + str(HERE / "frozen"),
            "-I" + str(HERE / "frozen/gemm_include/gfx950"),
            str(HERE / "layout_guard_check.cpp"), "-o", str(binary)]
    compiled = subprocess.run(argv, cwd=HERE, env=ENV, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (output / "layout_compile.log").write_text(compiled.stdout)
    if compiled.returncode:
        raise RuntimeError(compiled.stdout)
    checked = subprocess.check_output([str(binary)], cwd=HERE, env=ENV, text=True).strip()
    production_unchanged = all(sha(ROOT / row["original"]) == row["sha256"] for row in manifest["frozen_files"])
    assert production_unchanged and sha(Path(manifest["formal_control"]["path"])) == manifest["formal_control"]["sha256"]
    report = {"status": "cpu_audit_passed_unvalidated_numerics", "cpu_only": True,
              "gpu_operations": 0, "numerical_validation": "not_run_gpu_stopped",
              "performance_validation": "not_run_gpu_stopped", "libraries": records,
              "layout_guard_check": {"compile_argv": argv, "binary_sha256": sha(binary),
                                     "helper_source_sha256": sha(header), "stdout": checked,
                                     "host_attribute_adapter_sha256": sha(audit_opus / "opus.hpp"),
                                     "host_attribute_diff_sha256": sha(output / "host_attributes.diff")},
              "production_sources_unchanged": production_unchanged, "formal_control_unchanged": True,
              "elf_parser_reference": {"path": str(parser_path), "sha256": sha(parser_path)}}
    (HERE / "cpu_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
