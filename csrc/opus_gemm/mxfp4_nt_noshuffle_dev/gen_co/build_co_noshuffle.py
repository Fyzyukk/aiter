#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Offline .co builder for the no-shuffle a4w4_mxfp4_nt pipeline (pin toolchain), staged here until it is merged into
csrc/opus_gemm/gen_co.  Reuses gen_co/build_co_gfx950.py unchanged, pointed at this directory's
co_kernels_gfx950_noshuffle.json and gfx950/ output (images + build_info.json).

    python3 build_co_noshuffle.py --llvm-bin /root/toolchains/yuyzhang512-amdgpu-pin-op-dst-build/bin \\
        --opus-include /root/workspace/aiter-blockscale-mxfp8-nn-nt-tt/csrc/include
"""
import importlib.util, os
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("build_co_gfx950", os.path.join(HERE, "..", "..", "gen_co", "build_co_gfx950.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
m.JSON = os.path.join(HERE, "co_kernels_gfx950_noshuffle.json")
m.OUT_DIR = os.path.join(HERE, "gfx950")
m.load_entries.__defaults__ = (m.JSON,)
m.main()
