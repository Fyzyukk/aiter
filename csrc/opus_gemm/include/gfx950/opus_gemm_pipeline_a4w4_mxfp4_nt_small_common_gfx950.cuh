// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Device helpers shared by the small-tile MXFP4 x MXFP4 NT pipelines (gfx950), shuffle
// (opus_gemm_pipeline_a4w4_mxfp4_nt_small_gfx950.cuh) and row-major (..._nt_small_noshuffle_gfx950.cuh): vector types,
// static_for, buffer descriptors, split-K reduction over blocks, bf16 C store, per-stage LDS objects.
#pragma once
#include "opus_gemm_traits_a4w4_mxfp4_nt_small_common_gfx950.cuh"

#ifndef OPUS_MXFP4_SMALL_WAUX   // cache policy of the split-K partial-sum stores / loads (sc1)
#define OPUS_MXFP4_SMALL_WAUX 0x10
#endif

namespace opus_mxfp4_small {
using kargs_t = opus_gemm_mxfp4_small_kargs_gfx950;
typedef int   i32x4 __attribute__((ext_vector_type(4)));
typedef int   i32x8 __attribute__((ext_vector_type(8)));
typedef float f32x4 __attribute__((ext_vector_type(4)));
typedef unsigned u32x2 __attribute__((ext_vector_type(2)));
typedef unsigned u32x4 __attribute__((ext_vector_type(4)));
template<int I> using ic = std::integral_constant<int, I>;
template<int N, class F, int I = 0> __device__ __forceinline__ void static_for(F&& f) {
    if constexpr (I < N) { f(ic<I>{}); static_for<N, F, I + 1>(static_cast<F&&>(f)); }
}
__device__ __forceinline__ __amdgpu_buffer_rsrc_t rsrc(const void* base, unsigned bytes) {
    return __builtin_amdgcn_make_buffer_rsrc(const_cast<void*>(base), (short)0, (int)bytes, 0x00020000);
}
__device__ __forceinline__ unsigned clampb(long long v) { return v <= 0 ? 0u : (v > 0x7fffffffLL ? 0x7fffffffu : (unsigned)v); }
__device__ __forceinline__ unsigned bf16_rne(float f) {
    const unsigned u = __builtin_bit_cast(unsigned, f);
    return (u + 0x7fffu + ((u >> 16) & 1u)) >> 16;
}

// Split-K (S > 1): park the partial tile, the last of the S blocks to arrive adds the other S-1 partials.  Partials go
// through sc1 (write through to memory, read past the caches), so only "stores acknowledged before the counter atomic"
// is needed -- no L2 flush.  Returns false for the blocks that must not write C.
template<int EM, int EN, int BM, int BN>
__device__ __forceinline__ bool splitk_reduce(f32x4 (&acc)[EM][EN], const kargs_t& ka, bool owner, int lane,
                                              int row0, int wrow, int ncol0, int bz, int sk, int S) {
    const int Mp = (ka.m + BM - 1) / BM * BM, Np = (ka.n + BN - 1) / BN * BN;
    float* wsb = ka.ws + (long long)bz * S * Mp * Np;
    const auto rw = rsrc(wsb, clampb((long long)S * Mp * Np * 4)), r0w = rsrc(wsb, 0u);
    if (owner)
        static_for<EM>([&](auto e) { static_for<EN>([&](auto n) {
            const int o = ((sk * Mp + row0 + wrow + 16 * e() + lane % 16) * Np + ncol0 + 16 * n() + 4 * (lane / 16)) * 4;
            __builtin_amdgcn_raw_buffer_store_b128(__builtin_bit_cast(u32x4, acc[e()][n()]), rw, o, 0, OPUS_MXFP4_SMALL_WAUX);
        }); });
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    __syncthreads();
    __shared__ int s_last;
    int* ctr = ka.cnt + ((long long)bz * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
    if (threadIdx.x == 0) s_last = (__hip_atomic_fetch_add(ctr, 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) == S - 1);
    __syncthreads();
    if (!s_last) return false;
    if (owner)
        static_for<EM>([&](auto e) { static_for<EN>([&](auto n) {
            const int o0 = ((row0 + wrow + 16 * e() + lane % 16) * Np + ncol0 + 16 * n() + 4 * (lane / 16)) * 4;
            f32x4 part[7];
            static_for<7>([&](auto t) {          // up to 8 splits; splits >= S read 0 through a zero-range descriptor
                const int j = t() < sk ? t() : t() + 1;
                part[t()] = __builtin_bit_cast(f32x4, __builtin_amdgcn_raw_buffer_load_b128(t() < S - 1 ? rw : r0w,
                                               o0 + j * Mp * Np * 4, 0, OPUS_MXFP4_SMALL_WAUX));
            });
            static_for<7>([&](auto t) { acc[e()][n()] += part[t()]; });
        }); });
    if (threadIdx.x == 0) __hip_atomic_store(ctr, 0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);   // ready for the next launch
    return true;
}

// C: lane holds row wrow + 16e + lane%16 (relative to row0), columns ncol0 + 16n + 4*(lane/16) .. +3.
template<int EM, int EN>
__device__ __forceinline__ void store_c(const f32x4 (&acc)[EM][EN], const kargs_t& ka, int lane, int row0, int wrow, int ncol0, int bz) {
    uint16_t* pc = (uint16_t*)ka.ptr_c + bz * ka.stride_c_batch + (long long)row0 * ka.stride_c;
    const auto rc = rsrc(pc, clampb((long long)(ka.m - row0) * ka.stride_c * 2));
    static_for<EM>([&](auto e) { static_for<EN>([&](auto n) {
        const int col = ncol0 + 16 * n() + 4 * (lane / 16);
        if (col < ka.n) {
            const f32x4 v = acc[e()][n()];
            const u32x2 w = u32x2{bf16_rne(v[0]) | (bf16_rne(v[1]) << 16), bf16_rne(v[2]) | (bf16_rne(v[3]) << 16)};
            __builtin_amdgcn_raw_buffer_store_b64(__builtin_bit_cast(u32x2, w), rc,
                                                  ((wrow + 16 * e() + lane % 16) * ka.stride_c + col) * 2, 0, 0);
        }
    }); });
}

// One __shared__ object per ring stage: the compiler's waitcnt pass then knows an LDS-DMA into stage j cannot alias a
// ds_read of stage i, instead of waiting for every in-flight copy before each read (one array = everything aliases).
template<int I, int BYTES>
__device__ __forceinline__ char* lds_stage() {
    __shared__ __attribute__((aligned(16))) char buf[BYTES];
    return buf;
}

}   // namespace opus_mxfp4_small
