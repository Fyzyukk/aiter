#!/usr/bin/env python3
"""Reproduce frozen runtime headers checks without loading a GPU library.

Pass a new output directory. Existing artifacts are never overwritten. The
HIP driver only compiles objects or checks syntax; a plain C++ driver links
the CPU check, and its dynamic dependencies are checked before execution.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
ENV = dict(os.environ, HIP_VISIBLE_DEVICES="", ROCR_VISIBLE_DEVICES="", CUDA_VISIBLE_DEVICES="")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        raise SystemExit("Refusing to overwrite an existing output directory")
    out.mkdir(parents=True)
    frozen = HERE / "frozen"
    expected = json.loads((HERE / "source_inputs.json").read_text())
    for path, digest in expected.items():
        assert hashlib.sha256((frozen / path).read_bytes()).hexdigest() == digest, path
    shutil.copy2(HERE / "aggregate.hip", out / "aggregate.hip")
    shutil.copy2(HERE / "cpu_header_check.cpp", out / "cpu_header_check.cpp")
    shutil.copy2(HERE / "actual_b_layout_helpers.h", out / "actual_b_layout_helpers.h")
    shutil.copytree(HERE / "host_include", out / "host_include")
    records = []

    def run(label, command):
        proc = subprocess.run(command, env=ENV, text=True, capture_output=True)
        log = out / (label + ".log")
        log.write_text(proc.stdout + proc.stderr)
        records.append(dict(label=label, argv=command, returncode=proc.returncode,
                            log_sha256=hashlib.sha256(log.read_bytes()).hexdigest()))
        if proc.returncode:
            raise SystemExit(f"{label} failed; see {log}")
        return proc.stdout

    device = json.loads((HERE / "device_compile_final_argv.json").read_text())
    prefix = device[:device.index("--offload-device-only")]
    prefix = [arg for arg in prefix if not arg.startswith("-I")]
    prefix += ["-I" + str(frozen / "csrc/include"),
               "-I" + str(frozen / "csrc/opus_gemm/include"),
               "-I" + str(frozen / "csrc/opus_gemm/include/gfx950")]
    run("device_compile", prefix + ["--offload-device-only", "-c", str(out / "aggregate.hip"),
                                    "-o", str(out / "aggregate.o")])
    run("host_syntax", prefix + ["--offload-host-only", "-fsyntax-only", str(out / "aggregate.hip")])
    cpu = json.loads((HERE / "isolated_cpu_compile_argv.json").read_text())
    cpu_prefix = cpu[:cpu.index("-I" + str(HERE / "host_include"))]
    run("cpu_compile", cpu_prefix + ["-I" + str(out / "host_include"), "-I" + str(out),
                                    "-I" + str(frozen / "csrc/opus_gemm/include"),
                                    str(out / "cpu_header_check.cpp"), "-c", "-o", str(out / "cpu_check.o")])
    compiler = prefix[0]
    run("cpu_link", [compiler, str(out / "cpu_check.o"), "-o", str(out / "cpu_check")])
    dynamic = run("cpu_dynamic", [str(Path(compiler).parent / "llvm-readelf"), "--dynamic", str(out / "cpu_check")])
    assert not any(lib in dynamic for lib in ("libamdhip", "libhsa", "libcuda", "libtorch"))
    run("cpu_check", [str(out / "cpu_check")])
    (out / "receipt.json").write_text(json.dumps(dict(status="passed", commands=records,
        checked_no_gpu_library_dependencies=True, numerical_gpu_validation="not_run",
        performance_gpu_validation="not_run"), indent=2) + "\n")


if __name__ == "__main__":
    main()
