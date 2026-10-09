#!/usr/bin/env python3
"""Offline aggregate source/traits collision check. Never loads HIP or a device."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
INCLUDE = ROOT / "csrc/opus_gemm/include"
MAPPING = INCLUDE / "gfx950/opus_gemm_mxscale_bpreshuffle_promoted_variants.json"
REFERENCE = ROOT / "reports/opus_flydsl_all_20261009/main_variants/build_receipt.json"
ENV = dict(os.environ, ROCR_VISIBLE_DEVICES="", HIP_VISIBLE_DEVICES="", CUDA_VISIBLE_DEVICES="")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(argv, log):
    start = time.monotonic()
    with log.open("w") as f:
        proc = subprocess.run(argv, cwd=ROOT, env=ENV, stdout=f, stderr=subprocess.STDOUT)
    return dict(argv=argv, returncode=proc.returncode, seconds=time.monotonic()-start,
                log=str(log.relative_to(ROOT)), log_sha256=sha(log))


def source_check(variants):
    # Only public kernel/traits names, include wiring, helper scoping and reducer
    # reuse differ from the frozen audited candidate device bodies.
    helper_names = ["A0_AGPR_BASE", "A1_AGPR_BASE", "B0_AGPR_BASE", "B1_AGPR_BASE",
                    "A0_M0_STAGE_AGPR_BASE", "C11_TAIL_AGPR_BASE", "C01_AGPR_BASE",
                    "pinned_accumulator_array", "accumulator_pin_traits",
                    "load_operand_fragment_pinned", "load_operand_chunks_pinned", "mma_scale_group"]
    old_symbols = {
        "geometry": "opus_private_main_geometry_kernel",
        "pin_fixed": "gemm_a8w8_mxfp8_scale_kernel_private_fixed",
        "pad_pin_fixed": "gemm_a8w8_mxfp8_scale_4wave_256x256_padded_m_kernel_private_fixed",
        "shortk": "opus_private_shortk9022_kernel",
        "small_direct_b": "opus_private_small_a_lds_direct_b_kernel",
        "register_split": "opus_private_register_split4_kernel",
        "large_output_panel16": "opus_private_large9030_panel16_kernel",
        "large_output_direct_b": "opus_private_large9030_direct_b_kernel",
    }

    def body(text, symbol):
        # Skip the empty host stub and select the device definition with arguments.
        position = text.index("void " + symbol + "(")
        while text[text.index("{", position):text.index("{", position)+2] == "{}":
            position = text.index("void " + symbol + "(", position+1)
        begin = text.index("{", position)
        level = 1
        end = begin + 1
        while level:
            level += (text[end] == "{") - (text[end] == "}")
            end += 1
        return text[begin:end]

    checks = []
    for family, old_symbol in old_symbols.items():
        row = next(v for v in variants if v["family"] == family)
        old_path, new_path = ROOT / row["report_source"], INCLUDE / row["pipeline_header"]
        old, new = old_path.read_text(), new_path.read_text()
        # Derive the original symbol from the source in case report naming differs.
        if "void " + old_symbol + "(" not in old:
            import re
            candidates = re.findall(r"void\s+(\w+)\(", old)
            old_symbol = next(c for c in candidates if "private" in c and "kernel" in c)
        original_body = body(old, old_symbol)
        promoted_body = body(new, row["kernel"])
        if family in ("pin_fixed", "pad_pin_fixed"):
            for helper in helper_names:
                promoted_body = promoted_body.replace("opus_gemm_mxscale_bpreshuffle_pin_detail::" + helper, helper)
        assert original_body == promoted_body, family
        checks.append(dict(family=family, source=str(old_path.relative_to(ROOT)),
                           source_sha256=sha(old_path), promoted=str(new_path.relative_to(ROOT)),
                           promoted_sha256=sha(new_path), device_body_sha256=hashlib.sha256(original_body.encode()).hexdigest(),
                           status="byte_identical_after_public_symbol_and_helper_scope_normalization"))
    return checks


def main():
    receipt_path = HERE / "receipt.json"
    if receipt_path.exists():
        raise SystemExit("Refusing to overwrite a completed receipt")
    mapping = json.loads(MAPPING.read_text())
    variants = mapping["variants"]
    assert len(variants) == len({v["id"] for v in variants}) == 64
    for v in variants:
        assert (INCLUDE / v["traits_header"]).is_file(), v
        assert (INCLUDE / v["pipeline_header"]).is_file(), v
    pin = [v for v in variants if v["family"] in ("pin_fixed", "pad_pin_fixed")]
    nonpin = [v for v in variants if v not in pin]
    assert len(pin) == 11 and len(nonpin) == 53
    source_identity = source_check(variants)

    existing = sorted(p for p in (INCLUDE / "gfx950").glob("opus_gemm_pipeline_a8w8_mxscale_bpreshuffle*gfx950.cuh")
                      if p.name not in {Path(v["pipeline_header"]).name for v in variants})
    pinned_baseline = {"opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_4wave_gfx950.cuh",
                       "opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_4wave_256x256_padded_m_gfx950.cuh"}
    all_traits = sorted({v["traits_header"] for v in variants})
    inputs = {str(MAPPING.relative_to(ROOT)):sha(MAPPING), str(REFERENCE.relative_to(ROOT)):sha(REFERENCE)}
    for path in sorted(set(INCLUDE / v[key] for v in variants for key in ("pipeline_header", "traits_header")) | set(existing)):
        inputs[str(path.relative_to(ROOT))] = sha(path)
    for v in variants:
        if v["report_source"] != "existing production pipeline":
            inputs[v["report_source"]] = sha(ROOT / v["report_source"])
    (HERE / "inputs_before.json").write_text(json.dumps(inputs,indent=2)+"\n")
    reference = json.loads(REFERENCE.read_text())
    builds = []
    for side, selected in (("main23",nonpin),("pin24",pin)):
        folder = HERE / side
        folder.mkdir()
        pipelines = sorted({v["pipeline_header"] for v in (variants if side=="pin24" else nonpin)} |
                           {str(p.relative_to(INCLUDE)) for p in existing if side=="pin24" or p.name not in pinned_baseline})
        text = '#include <hip/hip_runtime.h>\n#include <cstdint>\n#define __HIPCC_RTC__ 1\n'
        text += "".join('#include "'+p+'"\n' for p in all_traits)
        text += "".join('#include "'+p+'"\n' for p in pipelines)
        for v in variants:
            text += f'using Traits_{v["id"]} = {v["traits"]};\n'
            text += f'static_assert(sizeof(Traits_{v["id"]}) > 0);\n'
        for v in selected:
            text += f'template __global__ void {v["kernel"]}<Traits_{v["id"]}>(opus_gemm_mxscale_bpreshuffle_kargs_gfx950);\n'
        source = folder / "aggregate.hip"
        source.write_text(text)
        previous = next(b for b in reference["builds"] if b["side"]==side)
        compiler = previous["compile_argv"][0]
        assert sha(compiler) == previous["compiler_sha256"]
        argv = previous["compile_argv"]
        flags = argv[1:next(i for i,arg in enumerate(argv) if arg.startswith("-I"))]
        includes = ["-I"+str(ROOT/"csrc/include"), "-I"+str(INCLUDE), "-I"+str(INCLUDE/"gfx950")]
        resource = [arg for arg in argv if arg.startswith("-resource-dir=")]
        device = folder / "aggregate.o"
        command = [compiler,*flags,*includes,*resource,"--offload-device-only","-c",str(source),"-o",str(device)]
        entry = dict(side=side, instantiated_ids=[v["id"] for v in selected],
                     trait_ids=[v["id"] for v in variants], included_pipeline_headers=pipelines,
                     included_traits_headers=all_traits, source_sha256=sha(source), compiler_sha256=sha(compiler),
                     compiler_version=previous["compiler_version"], exact_reference_codegen_flags=flags,
                     compile=run(command,folder/"compile.log"))
        if entry["compile"]["returncode"] == 0:
            entry["object"] = str(device.relative_to(ROOT))
            entry["object_sha256"] = sha(device)
            # CPU ELF/text inspection only, no library load or execution.
            for label,extra in (("headers",["-h"]),("notes",["--notes"]),("symbols",["--symbols"])):
                tool = str(Path(compiler).parent/"llvm-readelf")
                entry[label] = run([tool,*extra,str(device)],folder/(label+".txt"))
            host_command = [compiler,*flags,*includes,*resource,"--offload-host-only","-fsyntax-only",str(source)]
            entry["host_syntax"] = run(host_command,folder/"host_syntax.log")
        builds.append(entry)
        print(side, "compile",entry["compile"]["returncode"], "seconds",round(entry["compile"]["seconds"],3), flush=True)
    after = {path:sha(ROOT/path) for path in inputs}
    assert inputs == after, "Input changed during header check"
    passed = all(b["compile"]["returncode"]==0 and b["host_syntax"]["returncode"]==0 and
                 all(b[k]["returncode"]==0 for k in ("headers","notes","symbols")) for b in builds)
    receipt = dict(status="offline_aggregate_headers_passed" if passed else "offline_aggregate_headers_failed",
                   gpu_operations=0, hip_or_hsa_library_loads=0, device_execution=0,
                   numerical_validation="not_run_gpu_stopped", performance_validation="not_run_gpu_stopped",
                   promoted_variant_count=64, new_pipeline_header_count=8, grouped_traits_header_count=4,
                   source_identity=source_identity, mapping_sha256=sha(MAPPING),
                   inputs_unchanged=True, inputs=inputs, builds=builds, build_script_sha256=sha(__file__))
    receipt_path.write_text(json.dumps(receipt,indent=2)+"\n")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
