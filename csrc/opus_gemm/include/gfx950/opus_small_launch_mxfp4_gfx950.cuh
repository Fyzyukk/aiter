// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Host-only helper for the JIT-built small-tile a4w4 launchers (opus_gemm_pipeline_a4w4_mxfp4_nt_small_gfx950.cuh):
// per-stream split-K workspace = [int counters, fixed MAX_COUNTERS][fp32 partials].  Counters are zeroed when the buffer is
// (re)allocated; every launch's reducing block resets the counters it used, so the buffer is reused without re-zeroing
// (one stream at a time, as the kernel requires).
#pragma once
#include "aiter_hip_common.h"
#include <hip/hip_runtime.h>
#include <mutex>
#include <unordered_map>

namespace opus_gfx950_small {
constexpr long MAX_COUNTERS = 1L << 16;
inline void workspace(hipStream_t stream, long counters, long partial_floats, int*& cnt, float*& ws) {
    AITER_CHECK(counters <= MAX_COUNTERS, "a4w4_mxfp4_nt_small split-K: too many tiles (", counters, ")");
    struct Ws { void* p = nullptr; long bytes = 0; };
    static std::mutex mu;
    static std::unordered_map<hipStream_t, Ws> map;
    const long bytes = MAX_COUNTERS * 4 + partial_floats * 4;
    std::lock_guard<std::mutex> g(mu);
    Ws& w = map[stream];
    if (w.bytes < bytes) {
        if (w.p) { HIP_CALL(hipStreamSynchronize(stream)); HIP_CALL(hipFree(w.p)); }
        HIP_CALL(hipMalloc(&w.p, bytes));
        HIP_CALL(hipMemsetAsync(w.p, 0, MAX_COUNTERS * 4, stream));
        w.bytes = bytes;
    }
    cnt = static_cast<int*>(w.p);
    ws = reinterpret_cast<float*>(static_cast<char*>(w.p) + MAX_COUNTERS * 4);
}
} // namespace opus_gfx950_small
