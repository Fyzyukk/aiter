#!/usr/bin/env python3
"""Execute extracted real C++ fusion launchers with a CPU HIP launch recorder."""
import argparse
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    destination = args.output_dir.resolve()
    destination.mkdir(exist_ok=False)
    elf = module("fused_host_elf", ROOT / "reports/opus_resume_20261008/sfa_packed/audit_device.py")
    audit = module("fused_host_images", HERE / "audit_cache.py")
    environment = {**os.environ, "HIP_VISIBLE_DEVICES": "", "ROCR_VISIBLE_DEVICES": "",
                   "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"}
    rows = []
    for directory, argument_type in (("fused_register", "opus745_fused_kargs"),
                                     ("fused_lds", "opus745_fused_lds_kargs"),
                                     ("fused_wavek", "opus745_fused_wavek_kargs")):
        path = HERE / directory
        build = json.loads((path / "build.json").read_text())
        assert build["returncode"] == 0 and sha(path / "experiments.so") == build["library_sha256"]
        source = (path / "launch.hip").read_text()
        # Keep actual traits, argument headers and the complete host launcher.
        # Replace device function declarations only; the binary has no HIP/HSA.
        aliases = re.findall(r"using FusedTraits\d+ = .*?;", source, re.S)
        kernels = re.findall(r"void (fused(?:_lds|_wavek)?_kernel\d+)\(", source)
        assert len(aliases) == len(kernels) == len(build["configurations"])
        host = source[source.index("#if !defined(__HIP_DEVICE_COMPILE__)\ntemplate<class T>"):]
        image, _ = audit.device_image(elf, (path / "experiments.so").read_bytes())
        metadata = elf.Elf(image).metadata()["amdhsa.kernels"]
        assert len(metadata) == len(kernels)
        sizes = {k[".args"][0][".size"] for k in metadata}
        assert len(sizes) == 1 and all(k[".args"][0][".offset"] == 0 for k in metadata)
        size = next(iter(sizes))
        lines = ["#include <cassert>", "#include <cstdint>", "#include <cstdio>",
                 f'#include "{path / "fused_kargs.cuh"}"', *aliases]
        lines.extend(f"void {name}({argument_type}) {{}}" for name in kernels)
        lines.extend([host, f"static_assert(sizeof({argument_type}) == {size});",
                      f"static {argument_type} recorded;", "static dim3 grid, block;",
                      "static size_t dynamic_lds; static int calls; static const void* recorded_kernel;",
                      """extern "C" int hipLaunchKernel(const void* kernel, dim3 g, dim3 b,
    void** argv, size_t lds, void* stream) {
    assert(stream == reinterpret_cast<void*>(0x8000));
    grid=g;block=b;dynamic_lds=lds;recorded_kernel=kernel;calls++;
""", f"    recorded=*static_cast<{argument_type}*>(argv[0]);return 0;\n}}",
                      """static int call(int variant, int m, int n, int k, int split,
    uintptr_t a=0x1000, uintptr_t b=0x2000, uintptr_t sa=0x3000,
    uintptr_t sb=0x4000, uintptr_t c=0x5000, uintptr_t ws=0x6000, uintptr_t counters=0x7000) {
    return launch(variant,reinterpret_cast<void*>(a),reinterpret_cast<void*>(b),
      reinterpret_cast<void*>(sa),reinterpret_cast<void*>(sb),reinterpret_cast<void*>(c),
      reinterpret_cast<float*>(ws),reinterpret_cast<int*>(counters),m,n,k,split,
      reinterpret_cast<void*>(0x8000));
}
int main() { int valid=0, rejected=0;
"""])
        for config, kernel in zip(build["configurations"], kernels):
            i = config["id"]
            c = config["compile_params"]
            tm, tn = c["tile_m"], c["tile_n"]
            threads = c["wave_m"] * c["wave_n"] * c["wave_k"] * 64
            static_lds = next(k[".group_segment_fixed_size"] for k in metadata if k[".name"] == build["kernels"][i]["name"])
            for m, n, k, split in ((1, 128, 384, 3), (17, 256, 7168, 4),
                                  (c["max_m"], 128, 16384, 16), (16, 128, 128, 1)):
                tiles = (m + tm - 1) // tm * (n // tn)
                expected_grid = (tiles + 7) // 8 * 8 * split
                loops = (k // 128 + split - 1) // split
                lds = (min(loops, c["stages"]) * (tm + tn) // 8 * 1056
                       + (tm + (tn + 127) // 128) * loops) if directory == "fused_lds" else 0
                assert lds + static_lds <= 160 * 1024
                ws, counters = (0, 0) if split == 1 else (0x6000, 0x7000)
                lines.append(f"""{{ int before=calls;assert(call({i},{m},{n},{k},{split},
    0x1000,0x2000,0x3000,0x4000,0x5000,{ws},{counters})==0);assert(calls==before+1);
    assert(grid.x=={expected_grid} && grid.y==1 && grid.z==1);
    assert(block.x=={threads} && block.y==1 && block.z==1 && dynamic_lds=={lds});
    assert(recorded_kernel==reinterpret_cast<const void*>({kernel}));
    assert(recorded.m=={m} && recorded.n=={n} && recorded.k=={k} && recorded.batch==1);
    assert(recorded.split_k=={split} && recorded.stride_a=={k} && recorded.stride_b=={k});
    assert(recorded.stride_c=={n} && recorded.stride_sfa=={m} && recorded.stride_sfb=={k//128});
    assert(recorded.ptr_a==reinterpret_cast<void*>(0x1000));
    assert(recorded.ptr_b==reinterpret_cast<void*>(0x2000));
    assert(recorded.ptr_sfa==reinterpret_cast<void*>(0x3000));
    assert(recorded.ptr_sfb==reinterpret_cast<void*>(0x4000));
    assert(recorded.ptr_c==reinterpret_cast<void*>(0x5000));
    assert(recorded.ptr_workspace==reinterpret_cast<float*>({ws}));
    assert(recorded.ptr_counters==reinterpret_cast<int*>({counters}));valid++; }}""")
            rejected_cases = [(i, 0, 128, 384, 3, []), (i, c["max_m"] + 1, 128, 384, 3, []),
                              (i, 16, 127, 384, 3, []), (i, 16, 128, 129, 1, []),
                              (i, 16, 128, 384, 0, []), (i, 16, 128, 384, 4, []),
                              (i, 16, 128, 384, 3, [0]),
                              (i, 16, 128, 384, 3, [0x1001]),
                              (i, 16, 128, 384, 3, [0x1000, 0x2000, 0x3000, 0x4000, 0x5000, 0]),
                              (i, 16, 128, 384, 3, [0x1000, 0x2000, 0x3000, 0x4000, 0x5000, 0x6000, 0]),
                              (i, 16, 128, 384, 3, [0x1000, 0x2000, 0x3000, 0x4000, 0x5000, 0x6001, 0x7000]),
                              (i, 16, 128, 384, 3, [0x1000, 0x2000, 0x3000, 0x4000, 0x5000, 0x6000, 0x7001])]
            if directory == "fused_lds":
                rejected_cases += [(i, 16, 128, 384, 3, [0x1000, 0x2000, 0x3001]),
                                   (i, 16, 128, 384, 3, [0x1000, 0x2000, 0x3000, 0x4001])]
            for variant, m, n, k, split, pointers in rejected_cases:
                values = [str(v) for v in (variant, m, n, k, split)] + [hex(v) for v in pointers]
                lines.append("{int before=calls;assert(call(" + ",".join(values)
                             + ")==1);assert(calls==before);rejected++;}")
        lines.append(f"assert(call({len(kernels)},16,128,384,3)==1);rejected++;")
        lines.append('std::printf("%d %d\\n",valid,rejected);return 0;}')
        test_source = destination / (directory + ".cpp")
        test_source.write_text("\n".join(lines) + "\n")
        obj, binary = test_source.with_suffix(".o"), test_source.with_suffix("")
        compiler = build["argv"][0]
        compile_argv = [compiler, "-x", "hip", "--offload-host-only", "--rocm-path=/opt/rocm",
                        "--hip-path=/opt/rocm", "--offload-arch=gfx950", "-std=c++20", "-O2",
                        "-D__HIPCC_RTC__", "-DOPUS_ENABLE_RUNTIME_QUERY=0", "-Wno-deprecated-attributes",
                        "-I" + str(ROOT / "csrc/include"), "-I" + str(ROOT / "csrc/opus_gemm/include"),
                        "-c", str(test_source), "-o", str(obj)]
        compiled = subprocess.run(compile_argv, env=environment, capture_output=True, text=True)
        (destination / (directory + ".compile.log")).write_text(compiled.stdout + compiled.stderr)
        assert compiled.returncode == 0, compiled.stderr
        link_argv = [compiler, str(obj), "-o", str(binary), "-Wl,--no-undefined"]
        linked = subprocess.run(link_argv, env=environment, capture_output=True, text=True)
        assert linked.returncode == 0, linked.stderr
        dependencies = subprocess.check_output(["readelf", "-d", str(binary)], text=True)
        assert "libamdhip64" not in dependencies and "libhsa-runtime" not in dependencies
        run = subprocess.run([str(binary)], env=environment, capture_output=True, text=True)
        (destination / (directory + ".run.log")).write_text(run.stdout + run.stderr)
        assert run.returncode == 0, directory + ": " + run.stderr
        valid, rejected = map(int, run.stdout.split())
        rows.append({"family": directory, "valid_launch_plans": valid, "rejected_calls": rejected,
                     "device_argument_bytes": size, "host_argument_bytes": size,
                     "source_sha256": sha(path / "launch.hip"), "test_sha256": sha(test_source),
                     "library_sha256": build["library_sha256"], "binary_sha256": sha(binary),
                     "compile_argv": compile_argv, "link_argv": link_argv,
                     "HIP_HSA_dependencies": False})
    report = {"status": "passed", "recorded_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "configurations": sum(len(json.loads((HERE / r["family"] / "build.json").read_text())["configurations"]) for r in rows),
              "valid_launch_plans": sum(r["valid_launch_plans"] for r in rows),
              "rejected_calls": sum(r["rejected_calls"] for r in rows), "rows": rows,
              "gpu_contexts": 0, "GPU_kernel_launches": 0,
              "scope": "actual C++ host launcher and device argument extent; GPU numerical/performance pending"}
    (destination / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}))


if __name__ == "__main__":
    main()
