# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Per-device-TU compiler selection for MXFP8 candidates that pin AGPRs."""

import os
import subprocess
import sys
from pathlib import Path


def opus_compiler_commands_per_source():
    """Keep the baseline compiler except for the four explicit pin kernels."""
    if not os.environ.get("OPUS_BASELINE_HIP_CLANG_PATH"):
        return {}
    from csrc.opus_gemm.opus_gemm_common import (
        A8W8_BPRESHUFFLE_PIN_AGPR_KIDS,
        a8w8_mxscale_gemm_bpreshuffle_kernels_list,
    )

    compiler_dir = os.environ.get("OPUS_HIP_CLANG_PATH")
    if not compiler_dir or not Path(compiler_dir, "clang++").is_file():
        raise FileNotFoundError("Mixed OPUS builds require OPUS_HIP_CLANG_PATH")
    launcher = str(Path(__file__).with_name("opus_hip_compile.py"))
    command = [sys.executable, launcher, "--compiler", str(Path(compiler_dir, "clang++"))]
    resource_dir = os.environ.get("OPUS_HIP_RESOURCE_DIR")
    if resource_dir:
        command += ["--resource-dir", resource_dir]
    command += ["--"]
    return {
        f"{a8w8_mxscale_gemm_bpreshuffle_kernels_list[kid].name}_C*.device.cu": command
        for kid in sorted(A8W8_BPRESHUFFLE_PIN_AGPR_KIDS)
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Compile a pin-AGPR OPUS translation unit")
    parser.add_argument("--compiler", type=Path, required=True)
    parser.add_argument("--resource-dir", type=Path)
    parser.add_argument("flags", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    flags = args.flags[1:] if args.flags[:1] == ["--"] else args.flags
    flags = [flag for flag in flags if not flag.startswith("-resource-dir=")]
    rocm_root = os.environ.get("ROCM_PATH") or os.environ.get("ROCM_HOME") or "/opt/rocm"
    command = [str(args.compiler), "-x", "hip", f"--rocm-path={rocm_root}", f"--hip-path={rocm_root}"]
    if args.resource_dir:
        command.append(f"-resource-dir={args.resource_dir}")
    return subprocess.call(command + flags)


if __name__ == "__main__":
    raise SystemExit(main())
