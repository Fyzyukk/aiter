// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Runtime loader / launcher for the gfx950 pre-compiled opus_gemm code objects (gen_co/gfx950/*.co, built offline by
// gen_co/build_co_gfx950.py with the pin toolchain). Host pass only.
//
// The .co directory is OPUS_GEN_CO_DIR (environment variable, else the compile-time macro set by the JIT module);
// images live in <dir>/gfx950/<symbol>.co and contain one extern "C" kernel named <symbol>.
#pragma once
#if !defined(__HIP_DEVICE_COMPILE__) && !defined(__HIPCC_RTC__)
#include <hip/hip_runtime.h>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>
#include <mutex>
#include <unordered_map>
#include "aiter_hip_common.h"
#include "gfx950/opus_gemm_traits_a4w4_mxfp4_nt_gfx950.cuh"

#ifndef OPUS_GEN_CO_DIR
#define OPUS_GEN_CO_DIR ""
#endif

namespace opus_gfx950_co {
struct MxCoModule {
    std::vector<char> image;   // kept alive while the kernel is registered
    AiterAsmKernelFast kernel;

    explicit MxCoModule(const std::string& symbol) : image(read_co(symbol)), kernel(symbol.c_str(), image.data()) {}

private:
    static std::vector<char> read_co(const std::string& symbol) {
        const char* env = std::getenv("OPUS_GEN_CO_DIR");
        const std::string root = env && *env ? env : OPUS_GEN_CO_DIR;
        AITER_CHECK(!root.empty(), "opus gfx950 .co directory is not set (OPUS_GEN_CO_DIR)");
        const std::string path = root + "/gfx950/" + symbol + ".co";
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        AITER_CHECK(file.is_open(), "opus gfx950: cannot open code object ", path);
        const std::streamsize size = file.tellg();
        AITER_CHECK(size > 4, "opus gfx950: invalid code object size ", path);
        file.seekg(0, std::ios::beg);
        std::vector<char> bytes(static_cast<size_t>(size));
        AITER_CHECK(static_cast<bool>(file.read(bytes.data(), size)), "opus gfx950: short read ", path);
        AITER_CHECK(bytes[0] == 0x7f && bytes[1] == 'E' && bytes[2] == 'L' && bytes[3] == 'F',
                    "opus gfx950: not a bare ELF code object: ", path);
        return bytes;
    }
};

// Per-stream split-K workspace (grown on demand; the counter block is zeroed on allocation and kept at zero by the kernel).
// Allocation is not graph-capture safe: run the largest split-K shape once eagerly before capturing.
inline void* mxfp4_sk_workspace(hipStream_t stream, long bytes, long counter_bytes) {
    struct Ws { void* p = nullptr; long bytes = 0; };
    static std::mutex mu;
    static std::unordered_map<hipStream_t, Ws> map;
    std::lock_guard<std::mutex> g(mu);
    Ws& w = map[stream];
    if (w.bytes < bytes) {
        if (w.p) { HIP_CALL(hipStreamSynchronize(stream)); HIP_CALL(hipFree(w.p)); }
        HIP_CALL(hipMalloc(&w.p, bytes));
        HIP_CALL(hipMemsetAsync(w.p, 0, counter_bytes, stream));
        w.bytes = bytes;
    }
    return w.p;
}

inline AiterAsmKernelFast& mxfp4_co_kernel(const std::string& symbol) {
    static SynchronizedCache<std::string, MxCoModule> cache;
    return cache.get_or_create(symbol, [&]() { return MxCoModule(symbol); }).kernel;
}
} // namespace opus_gfx950_co

// Split-K factor: largest S in {8, 4, 2} with tiles * S <= CUs and K a multiple of S * 256 (each segment whole K tiles);
// 1 if none (or tiles > OPUS_MXFP4_SK_MAX_TILES).  OPUS_MXFP4_SPLITK=<S> overrides (tuning / tests).
inline int opus_mxfp4_pick_split(int tiles, int K, int cus) {
    if (const char* e = std::getenv("OPUS_MXFP4_SPLITK")) {
        const int s = std::atoi(e);
        AITER_CHECK(s >= 1 && K % (s * 256) == 0, "OPUS_MXFP4_SPLITK=", s, " needs K % (S * 256) == 0 (K=", K, ")");
        return s;
    }
    if (tiles > OPUS_MXFP4_SK_MAX_TILES) return 1;
    for (int s : {8, 4, 2})
        if (tiles * s <= cus && K % (s * 256) == 0) return s;
    return 1;
}

template <class Kargs>
inline void opus_mxfp4_co_launch_gfx950(AiterAsmKernelFast& kernel, dim3 grid, dim3 block, Kargs& args, hipStream_t stream) {
    size_t arg_size = sizeof(args);
    kernel.launch_kernel({&args, &arg_size, int(grid.x), int(grid.y), int(grid.z), int(block.x), int(block.y), int(block.z), stream});
}
#endif
