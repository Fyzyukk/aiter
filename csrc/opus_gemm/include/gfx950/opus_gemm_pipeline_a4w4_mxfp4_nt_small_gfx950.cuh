// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Small-tile MXFP4 x MXFP4 NT GEMM (gfx950): one file for the small-M (M <= ~64) and medium-M (M ~ 128 - 1024) tiles.
// Standard ROCm toolchain: no pin attributes, no opus (HIP builtins only).  LAYOUT: inputs as aiter gemm_a4w4(bpreshuffle=True)
// (SHUFFLED) or as aiter Triton gemm_afp4wfp4 (ROWMAJOR: B and scales row-major, no padding).
//
//   B_DIRECT (BMODE 0): every wave streams its A / B / scale operands from global straight into registers through a D-deep
//     ring (no LDS, no barrier in the K loop).  KS splits the 4 waves along K (partials summed through LDS once at the end).
//   B_LDS    (BMODE 1): A and B go global -> LDS by LDS-DMA (buffer_load ... lds) through a D-deep LDS ring, one barrier per
//     K tile; scales go straight into registers.  1x4 waves (BM = 32 / 64) or 2x2 waves (BM = 128).
// Both: MFMA v_mfma_scale_f32_16x16x128_f8f6f4 with B as srcA (each lane holds 4 consecutive columns of one C row);
//   scales read as raw e8m0_shuffle dwords ([kk0 half0, kk0 half1, kk1 half0, kk1 half1] of a 32-row block) with a
//   compile-time opsel; split-K over blocks (last-arriving block reduces); M / N tails by buffer ranges + 16-column C mask.
#pragma once
#include "opus_gemm_traits_a4w4_mxfp4_nt_small_gfx950.cuh"

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

// Row-major scales: for a 16-row block, lane L needs row L%16, K groups g + 4kk (g = L/16, kk = 0/1) of the K tile:
// one 8-byte load of the row's 8 groups, then one v_perm moves byte g -> byte 0 (kk 0) and byte 4+g -> byte 1 (kk 1),
// so the MFMA picks them with opsel = kk.
__device__ __forceinline__ unsigned rm_sel(int lane) { const unsigned g = lane / 16; return g | ((4u + g) << 8) | 0x0c0c0000u; }
__device__ __forceinline__ unsigned rm_pack(u32x2 v, unsigned sel) { return __builtin_amdgcn_perm(v[1], v[0], sel); }

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

// ---------------------------------------------------------------------------------------------------------------------
// B_DIRECT: no LDS in the K loop.  KS: the 4 waves split KS ways along K (wave group kg takes K tiles kg, kg + KS, ...)
// and 4 / KS ways along N; the KS partial accumulators are summed through LDS at the end (one barrier).
template<int BM, int BN, int D, int KS>
__device__ __forceinline__ void body_direct(const kargs_t& ka) {
    static_assert(BM == 16 || BM == 32 || BM == 64 || BM == 128);
    static_assert(BN == 32 || BN == 64 || BN == 128 || BN == 256);
    constexpr int EM = BM / 16;
    static_assert(KS == 1 || KS == 2 || KS == 4);
    constexpr int WN = 4 / KS;
    constexpr int EN = BN / (16 * WN);
    static_assert(EN % 2 == 0, "a wave's columns must cover whole 32-row scale blocks");
    constexpr int SA = (BM + 31) / 32, SB = EN / 2;
    const int lane = __lane_id();
    const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x / 64);
    const int S = ka.splitk;
    const int col0 = blockIdx.x * BN, row0 = blockIdx.y * BM, bz = blockIdx.z / S, sk = blockIdx.z % S;
    const int ktot = ka.k / 256, kper = (ktot + S - 1) / S, kbeg = sk * kper;
    const int loops = (ktot < kbeg + kper ? ktot : kbeg + kper);    // this block's K tiles: [kbeg, loops)
    const int Kb = ka.k / 2;
    const int wn = wave % WN, kgrp = wave / WN;
    const int ncol0 = col0 + wn * (BN / WN);

    const uint8_t* pa  = (const uint8_t*)ka.ptr_a   + bz * ka.stride_a_batch   + (long long)row0 * ka.stride_a;
    const uint8_t* pb  = (const uint8_t*)ka.ptr_b   + bz * ka.stride_b_batch   + (long long)ncol0 * Kb;
    const uint8_t* psa = (const uint8_t*)ka.ptr_sfa + bz * ka.stride_sfa_batch + (long long)(row0 / 32) * ka.stride_sfa;
    const uint8_t* psb = (const uint8_t*)ka.ptr_sfb + bz * ka.stride_sfb_batch + (long long)(ncol0 / 32) * ka.stride_sfb;
    const auto ra_  = rsrc(pa,  clampb((long long)(ka.m - row0) * ka.stride_a));
    const auto rb_  = rsrc(pb,  clampb((long long)(ka.n - ncol0) * Kb));
    const auto rsa_ = rsrc(psa, clampb((long long)((ka.m + 31) / 32 - row0 / 32) * ka.stride_sfa));
    const auto rsb_ = rsrc(psb, clampb((long long)((ka.n + 31) / 32 - ncol0 / 32) * ka.stride_sfb));
    const int a_lane = (lane % 16) * ka.stride_a + (lane / 16) * 16;   // A operand: row lane%16, K bytes [16*(lane/16), +16)
    const unsigned sh_a = (BM == 16) ? 8u * ((row0 / 16) & 1) : 0u;    // BM == 16: half of the 32-row scale block

    i32x4 va[D][EM][2], vb[D][EN][2];
    unsigned vsa[D][SA], vsb[D][SB];
    // K tiles past the end load through zero-range descriptors (reads return 0, scales too), so the MFMAs run
    // unconditionally: no branch in the loop keeps the compiler's vmcnt counting exact across the ring.
    const auto r0 = rsrc(pa, 0u);
    auto issue = [&](auto s_, int kt) {
        constexpr int s = decltype(s_)::value;
        const bool in = kt < loops;
        const auto ra = in ? ra_ : r0;  const auto rb = in ? rb_ : r0;
        const auto rsa = in ? rsa_ : r0; const auto rsb = in ? rsb_ : r0;
        static_for<EN>([&](auto e) { static_for<2>([&](auto kk) {
            vb[s][e()][kk()] = __builtin_bit_cast(i32x4, __builtin_amdgcn_raw_buffer_load_b128(rb, lane * 16, e() * 16 * Kb + kt * 2048 + kk() * 1024, 0));
        }); });
        static_for<EM>([&](auto e) { static_for<2>([&](auto kk) {
            va[s][e()][kk()] = __builtin_bit_cast(i32x4, __builtin_amdgcn_raw_buffer_load_b128(ra, a_lane, e() * 16 * ka.stride_a + kt * 128 + kk() * 64, 0));
        }); });
        static_for<SA>([&](auto j) { vsa[s][j()] = __builtin_amdgcn_raw_buffer_load_b32(rsa, lane * 4, j() * ka.stride_sfa + kt * 256, 0); });
        static_for<SB>([&](auto j) { vsb[s][j()] = __builtin_amdgcn_raw_buffer_load_b32(rsb, lane * 4, j() * ka.stride_sfb + kt * 256, 0); });
    };

    f32x4 acc[EM][EN];
    static_for<EM>([&](auto e) { static_for<EN>([&](auto n) { acc[e()][n()] = f32x4{0.f, 0.f, 0.f, 0.f}; }); });
    auto compute = [&](auto s_) {
        constexpr int s = decltype(s_)::value;
        static_for<2>([&](auto kk) { static_for<EM>([&](auto e) { static_for<EN>([&](auto n) {
            const i32x8 a8 = __builtin_shufflevector(va[s][e()][kk()], va[s][e()][kk()], 0, 1, 2, 3, -1, -1, -1, -1);
            const i32x8 b8 = __builtin_shufflevector(vb[s][n()][kk()], vb[s][n()][kk()], 0, 1, 2, 3, -1, -1, -1, -1);
            constexpr int ja = (BM == 16) ? 0 : e() / 2, opa = 2 * kk() + ((BM == 16) ? 0 : e() % 2);
            constexpr int jb = n() / 2, opb = 2 * kk() + n() % 2;
            const unsigned sa = (BM == 16) ? (vsa[s][ja] >> sh_a) : vsa[s][ja];
            acc[e()][n()] = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(b8, a8, acc[e()][n()], 4, 4,
                                                                             opb, (int)vsb[s][jb], opa, (int)sa);
        }); }); });
    };
    const int k0 = kbeg + kgrp;
    const int nk = loops > k0 ? (loops - k0 + KS - 1) / KS : 0;
    static_for<D>([&](auto s) { issue(s, k0 + KS * s()); __builtin_amdgcn_sched_barrier(0); });   // same order as the loop
#pragma clang loop unroll(disable)
    for (int i0 = 0; i0 < nk; i0 += D) {
        static_for<D>([&](auto s) {
            const int i = i0 + s();
            compute(s);                          // i >= nk: all operands are 0 (zero-range loads), adds nothing
            __builtin_amdgcn_sched_barrier(0);   // keep the ring order: stage s computes, then reloads
            issue(s, k0 + KS * (i + D));
            __builtin_amdgcn_sched_barrier(0);
        });
    }
    if constexpr (KS > 1) {   // groups 1..KS-1 park their partials in LDS, group 0 adds them
        constexpr int FR = EM * EN;
        __shared__ f32x4 red[(KS - 1) * WN * FR * 64];
        if (kgrp > 0)
            static_for<EM>([&](auto e) { static_for<EN>([&](auto n) {
                red[(((kgrp - 1) * WN + wn) * FR + e() * EN + n()) * 64 + lane] = acc[e()][n()];
            }); });
        __syncthreads();
        if (kgrp == 0)
            static_for<KS - 1>([&](auto g) { static_for<EM>([&](auto e) { static_for<EN>([&](auto n) {
                acc[e()][n()] += red[((g() * WN + wn) * FR + e() * EN + n()) * 64 + lane];
            }); }); });
    }
    const bool owner = (KS == 1) || (kgrp == 0);
    if (S > 1 && !splitk_reduce<EM, EN, BM, BN>(acc, ka, owner, lane, row0, 0, ncol0, bz, sk, S)) return;
    if (!owner) return;
    store_c<EM, EN>(acc, ka, lane, row0, 0, ncol0, bz);
}

// ---------------------------------------------------------------------------------------------------------------------
// Row-major layout (Triton gemm_afp4wfp4 inputs): separate copies of the two bodies so that the shuffled kernels above
// keep exactly their generated code.  They are written for both layouts (LAYOUT template) but only instantiated with
// ROWMAJOR.
// ---------------------------------------------------------------------------------------------------------------------
// B_DIRECT: no LDS in the K loop.  KS: the 4 waves split KS ways along K (wave group kg takes K tiles kg, kg + KS, ...)
// and 4 / KS ways along N; the KS partial accumulators are summed through LDS at the end (one barrier).
template<int BM, int BN, int D, int KS, int LAYOUT>
__device__ __forceinline__ void body_direct_rm(const kargs_t& ka) {
    constexpr bool RM = LAYOUT == OPUS_MXFP4_SMALL_ROWMAJOR;
    static_assert(BM == 16 || BM == 32 || BM == 64 || BM == 128);
    static_assert(BN == 32 || BN == 64 || BN == 128 || BN == 256);
    constexpr int EM = BM / 16;
    static_assert(KS == 1 || KS == 2 || KS == 4);
    constexpr int WN = 4 / KS;
    constexpr int EN = BN / (16 * WN);
    static_assert(RM || EN % 2 == 0, "a wave's columns must cover whole 32-row scale blocks");
    constexpr int SA = RM ? EM : (BM + 31) / 32, SB = RM ? EN : EN / 2;   // scale loads per K tile (RM: one per 16-row block)
    const int lane = __lane_id();
    const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x / 64);
    const int S = ka.splitk;
    const int col0 = blockIdx.x * BN, row0 = blockIdx.y * BM, bz = blockIdx.z / S, sk = blockIdx.z % S;
    const int ktot = ka.k / 256, kper = (ktot + S - 1) / S, kbeg = sk * kper;
    const int loops = (ktot < kbeg + kper ? ktot : kbeg + kper);    // this block's K tiles: [kbeg, loops)
    const int Kb = ka.k / 2;
    const int wn = wave % WN, kgrp = wave / WN;
    const int ncol0 = col0 + wn * (BN / WN);

    const uint8_t* pa  = (const uint8_t*)ka.ptr_a   + bz * ka.stride_a_batch   + (long long)row0 * ka.stride_a;
    const uint8_t* pb  = (const uint8_t*)ka.ptr_b   + bz * ka.stride_b_batch   + (long long)ncol0 * Kb;
    const uint8_t* psa = (const uint8_t*)ka.ptr_sfa + bz * ka.stride_sfa_batch + (long long)(RM ? row0 : row0 / 32) * ka.stride_sfa;
    const uint8_t* psb = (const uint8_t*)ka.ptr_sfb + bz * ka.stride_sfb_batch + (long long)(RM ? ncol0 : ncol0 / 32) * ka.stride_sfb;
    const auto ra_  = rsrc(pa,  clampb((long long)(ka.m - row0) * ka.stride_a));
    const auto rb_  = rsrc(pb,  clampb((long long)(ka.n - ncol0) * Kb));
    const auto rsa_ = rsrc(psa, clampb(RM ? (long long)(ka.m - row0) * ka.stride_sfa : (long long)((ka.m + 31) / 32 - row0 / 32) * ka.stride_sfa));
    const auto rsb_ = rsrc(psb, clampb(RM ? (long long)(ka.n - ncol0) * ka.stride_sfb : (long long)((ka.n + 31) / 32 - ncol0 / 32) * ka.stride_sfb));
    const int a_lane = (lane % 16) * ka.stride_a + (lane / 16) * 16;   // A operand: row lane%16, K bytes [16*(lane/16), +16)
    const int b_lane = RM ? (lane % 16) * Kb + (lane / 16) * 16 : lane * 16;   // RM: B operand addressed like A
    const unsigned sh_a = (!RM && BM == 16) ? 8u * ((row0 / 16) & 1) : 0u;    // shuffled, BM == 16: half of the 32-row scale block
    const int sa_lane = RM ? (lane % 16) * ka.stride_sfa : lane * 4;
    const int sb_lane = RM ? (lane % 16) * ka.stride_sfb : lane * 4;
    const unsigned rsel = rm_sel(lane);

    // K tiles of this wave.  Shuffled: interleaved (kt = kbeg + kgrp + KS*i).  Row-major: a contiguous chunk per wave
    // group, so with an even ring depth (PAIR) the row-major scales of tiles (kt, kt+1) -- 16 contiguous bytes per row --
    // come in with one 16-byte load at the odd stage (half the scale instructions, each touching 16 rows).
    constexpr bool PAIR = RM && (D % 2 == 0);
    constexpr int KSTEP = RM ? 1 : KS;
    const int kchunk = RM ? ((loops - kbeg) + KS - 1) / KS : 0;
    const int k0 = RM ? kbeg + kgrp * kchunk : kbeg + kgrp;
    const int kend_w = RM ? (loops < k0 + kchunk ? loops : k0 + kchunk) : loops;
    const int nk = kend_w > k0 ? (kend_w - k0 + KSTEP - 1) / KSTEP : 0;

    i32x4 va[D][EM][2], vb[D][EN][2];
    using sf_t = std::conditional_t<RM, u32x2, unsigned>;
    sf_t vsa[D][SA], vsb[D][SB];
    u32x4 psa2[PAIR ? D / 2 : 1][PAIR ? SA : 1], psb2[PAIR ? D / 2 : 1][PAIR ? SB : 1];
    // K tiles past the end load through zero-range descriptors (reads return 0, scales too), so the MFMAs run
    // unconditionally: no branch in the loop keeps the compiler's vmcnt counting exact across the ring.
    const auto r0 = rsrc(pa, 0u);
    auto issue = [&](auto s_, int kt) {
        constexpr int s = decltype(s_)::value;
        const bool in = kt < kend_w;
        const auto ra = in ? ra_ : r0;  const auto rb = in ? rb_ : r0;
        const auto rsa = in ? rsa_ : r0; const auto rsb = in ? rsb_ : r0;
        static_for<EN>([&](auto e) { static_for<2>([&](auto kk) {
            const int ob = RM ? e() * 16 * Kb + kt * 128 + kk() * 64 : e() * 16 * Kb + kt * 2048 + kk() * 1024;
            vb[s][e()][kk()] = __builtin_bit_cast(i32x4, __builtin_amdgcn_raw_buffer_load_b128(rb, b_lane, ob, 0));
        }); });
        static_for<EM>([&](auto e) { static_for<2>([&](auto kk) {
            va[s][e()][kk()] = __builtin_bit_cast(i32x4, __builtin_amdgcn_raw_buffer_load_b128(ra, a_lane, e() * 16 * ka.stride_a + kt * 128 + kk() * 64, 0));
        }); });
        if constexpr (RM) {
#ifdef SMALL_EXP_NOSCALE   // experiment only: skip the scale loads (results wrong)
            static_for<SA>([&](auto j) { vsa[s][j()] = u32x2{0x7f7f7f7fu, 0x7f7f7f7fu}; });
            static_for<SB>([&](auto j) { vsb[s][j()] = u32x2{0x7f7f7f7fu, 0x7f7f7f7fu}; });
#else
            if constexpr (PAIR) {
                if constexpr (s % 2 == 1) {   // tiles (kt-1, kt): stage s-1 was refilled just before, both now free
                    const bool in2 = kt - 1 < kend_w, hi = kt < kend_w;
                    const auto rsa2 = in2 ? rsa_ : r0; const auto rsb2 = in2 ? rsb_ : r0;
                    static_for<SA>([&](auto j) {
                        u32x4 v = __builtin_bit_cast(u32x4, __builtin_amdgcn_raw_buffer_load_b128(rsa2, sa_lane, j() * 16 * ka.stride_sfa + (kt - 1) * 8, 0));
                        if (!hi) { v[2] = 0; v[3] = 0; }   // tile kt past this wave's range: scale 0 (never padding / next-row bytes)
                        psa2[s / 2][j()] = v;
                    });
                    static_for<SB>([&](auto j) {
                        u32x4 v = __builtin_bit_cast(u32x4, __builtin_amdgcn_raw_buffer_load_b128(rsb2, sb_lane, j() * 16 * ka.stride_sfb + (kt - 1) * 8, 0));
                        if (!hi) { v[2] = 0; v[3] = 0; }
                        psb2[s / 2][j()] = v;
                    });
                }
            } else {
                static_for<SA>([&](auto j) { vsa[s][j()] = __builtin_bit_cast(u32x2, __builtin_amdgcn_raw_buffer_load_b64(rsa, sa_lane, j() * 16 * ka.stride_sfa + kt * 8, 0)); });
                static_for<SB>([&](auto j) { vsb[s][j()] = __builtin_bit_cast(u32x2, __builtin_amdgcn_raw_buffer_load_b64(rsb, sb_lane, j() * 16 * ka.stride_sfb + kt * 8, 0)); });
            }
#endif
        } else {
            static_for<SA>([&](auto j) { vsa[s][j()] = __builtin_amdgcn_raw_buffer_load_b32(rsa, sa_lane, j() * ka.stride_sfa + kt * 256, 0); });
            static_for<SB>([&](auto j) { vsb[s][j()] = __builtin_amdgcn_raw_buffer_load_b32(rsb, sb_lane, j() * ka.stride_sfb + kt * 256, 0); });
        }
    };

    f32x4 acc[EM][EN];
    static_for<EM>([&](auto e) { static_for<EN>([&](auto n) { acc[e()][n()] = f32x4{0.f, 0.f, 0.f, 0.f}; }); });
    auto compute = [&](auto s_) {
        constexpr int s = decltype(s_)::value;
        unsigned pa_[RM ? EM : 1], pb_[RM ? EN : 1];
        if constexpr (PAIR) {
            static_for<EM>([&](auto e) { const u32x4 v = psa2[s / 2][e()]; pa_[e()] = rm_pack(s % 2 ? u32x2{v[2], v[3]} : u32x2{v[0], v[1]}, rsel); });
            static_for<EN>([&](auto n) { const u32x4 v = psb2[s / 2][n()]; pb_[n()] = rm_pack(s % 2 ? u32x2{v[2], v[3]} : u32x2{v[0], v[1]}, rsel); });
        } else if constexpr (RM) {
            static_for<EM>([&](auto e) { pa_[e()] = rm_pack(vsa[s][e()], rsel); });
            static_for<EN>([&](auto n) { pb_[n()] = rm_pack(vsb[s][n()], rsel); });
        }
        static_for<2>([&](auto kk) { static_for<EM>([&](auto e) { static_for<EN>([&](auto n) {
            const i32x8 a8 = __builtin_shufflevector(va[s][e()][kk()], va[s][e()][kk()], 0, 1, 2, 3, -1, -1, -1, -1);
            const i32x8 b8 = __builtin_shufflevector(vb[s][n()][kk()], vb[s][n()][kk()], 0, 1, 2, 3, -1, -1, -1, -1);
            if constexpr (RM) {
                acc[e()][n()] = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(b8, a8, acc[e()][n()], 4, 4,
                                                                                 kk(), (int)pb_[n()], kk(), (int)pa_[e()]);
            } else {
                constexpr int ja = (BM == 16) ? 0 : e() / 2, opa = 2 * kk() + ((BM == 16) ? 0 : e() % 2);
                constexpr int jb = n() / 2, opb = 2 * kk() + n() % 2;
                const unsigned sa = (BM == 16) ? (vsa[s][ja] >> sh_a) : vsa[s][ja];
                acc[e()][n()] = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(b8, a8, acc[e()][n()], 4, 4,
                                                                                 opb, (int)vsb[s][jb], opa, (int)sa);
            }
        }); }); });
    };
    static_for<D>([&](auto s) { issue(s, k0 + KSTEP * s()); __builtin_amdgcn_sched_barrier(0); });   // same order as the loop
#pragma clang loop unroll(disable)
    for (int i0 = 0; i0 < nk; i0 += D) {
        static_for<D>([&](auto s) {
            const int i = i0 + s();
            compute(s);                          // i >= nk: all operands are 0 (zero-range loads), adds nothing
            __builtin_amdgcn_sched_barrier(0);   // keep the ring order: stage s computes, then reloads
            issue(s, k0 + KSTEP * (i + D));
            __builtin_amdgcn_sched_barrier(0);
        });
    }
    if constexpr (KS > 1) {   // groups 1..KS-1 park their partials in LDS, group 0 adds them
        constexpr int FR = EM * EN;
        __shared__ f32x4 red[(KS - 1) * WN * FR * 64];
        if (kgrp > 0)
            static_for<EM>([&](auto e) { static_for<EN>([&](auto n) {
                red[(((kgrp - 1) * WN + wn) * FR + e() * EN + n()) * 64 + lane] = acc[e()][n()];
            }); });
        __syncthreads();
        if (kgrp == 0)
            static_for<KS - 1>([&](auto g) { static_for<EM>([&](auto e) { static_for<EN>([&](auto n) {
                acc[e()][n()] += red[((g() * WN + wn) * FR + e() * EN + n()) * 64 + lane];
            }); }); });
    }
    const bool owner = (KS == 1) || (kgrp == 0);
    if (S > 1 && !splitk_reduce<EM, EN, BM, BN>(acc, ka, owner, lane, row0, 0, ncol0, bz, sk, S)) return;
    if (!owner) return;
    store_c<EM, EN>(acc, ka, lane, row0, 0, ncol0, bz);
}

// One __shared__ object per ring stage: the compiler's waitcnt pass then knows an LDS-DMA into stage j cannot alias a
// ds_read of stage i, instead of waiting for every in-flight copy before each read (one array = everything aliases).
template<int I, int BYTES>
__device__ __forceinline__ char* lds_stage() {
    __shared__ __attribute__((aligned(16))) char buf[BYTES];
    return buf;
}

// ---------------------------------------------------------------------------------------------------------------------
// B_LDS: A and B global -> LDS by LDS-DMA through a D-deep ring of LDS stages (one barrier per K tile), scales straight
// into registers.  Stage image: A = BM/8 blocks of 8 rows (one DMA = 64 lanes x 16 B), block stride 1088 B, row r of a
// block at r*128 with its 16 B chunk c at slot c ^ ((r >> 1) & 3) (the 16 lanes of an A ds_read_b128 hit distinct bank
// groups); B = the shuffle_weight image copied verbatim: 16-column block jb, K half kk = 1 KB at jb*2048 + kk*1024, read
// lane*16 (contiguous, conflict free).  The A / B DMAs of a K tile are spread round-robin over the 4 waves.
// Ring: K tile kt lives in stage (kt - kbeg) % D; iteration kt waits for its own copies (counted vmcnt: exactly D-2 younger
// copy groups are in flight), passes the barrier, refills the stage of K tile kt-1 with K tile kt+D-1, then computes kt.
template<int BM, int BN, int D>
__device__ __forceinline__ void body_lds(const kargs_t& ka) {
    static_assert(BM == 32 || BM == 64 || BM == 128);
    static_assert(BN == 128 || BN == 256);
    constexpr int WM = (BM == 128) ? 2 : 1, WN = 4 / WM;           // 1x4 for BM <= 64, 2x2 for BM == 128
    static_assert(!(BM == 128 && BN != 128), "BM = 128 is 128 x 128 (2x2 waves)");
    constexpr int WROWS = BM / WM, WCOLS = BN / WN;
    constexpr int EM = WROWS / 16, EN = WCOLS / 16;
    static_assert(EM % 2 == 0 && EN % 2 == 0, "waves cover whole 32-row scale blocks");
    constexpr int SA = WROWS / 32, SB = WCOLS / 32;
    constexpr int BLK = 1088;
    constexpr int A_BYTES = BM / 8 * BLK, B_BYTES = BN * 128, STAGE = A_BYTES + B_BYTES;
    static_assert(D * STAGE <= 160 * 1024, "LDS ring too deep for this tile");
    constexpr int A_DMA = BM / 8 / 4, B_DMA = BN / 16 * 2 / 4;       // per wave per K tile
    static_assert(A_DMA >= 1 && B_DMA >= 1);
    constexpr int G = A_DMA + B_DMA + SA + SB;                        // VMEM instructions per wave per K tile
    static_assert((D - 2) * G <= 63, "vmcnt range");

    const int lane = __lane_id();
    const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x / 64);
    const int wm = wave / WN, wn = wave % WN;
    const int S = ka.splitk;
    const int col0 = blockIdx.x * BN, row0 = blockIdx.y * BM, bz = blockIdx.z / S, sk = blockIdx.z % S;
    const int ktot = ka.k / 256, kper = (ktot + S - 1) / S, kbeg = sk * kper;
    const int kend = (ktot < kbeg + kper ? ktot : kbeg + kper);
    const int Kb = ka.k / 2;
    const int wrow = wm * WROWS, ncol0 = col0 + wn * WCOLS;

    const uint8_t* pa  = (const uint8_t*)ka.ptr_a   + bz * ka.stride_a_batch   + (long long)row0 * ka.stride_a;
    const uint8_t* pb  = (const uint8_t*)ka.ptr_b   + bz * ka.stride_b_batch   + (long long)col0 * Kb;
    const uint8_t* psa = (const uint8_t*)ka.ptr_sfa + bz * ka.stride_sfa_batch + (long long)((row0 + wrow) / 32) * ka.stride_sfa;
    const uint8_t* psb = (const uint8_t*)ka.ptr_sfb + bz * ka.stride_sfb_batch + (long long)(ncol0 / 32) * ka.stride_sfb;
    const auto ra_  = rsrc(pa,  clampb((long long)(ka.m - row0) * ka.stride_a));
    const auto rb_  = rsrc(pb,  clampb((long long)(ka.n - col0) * Kb));
    const auto rsa_ = rsrc(psa, clampb((long long)((ka.m + 31) / 32 - (row0 + wrow) / 32) * ka.stride_sfa));
    const auto rsb_ = rsrc(psb, clampb((long long)((ka.n + 31) / 32 - ncol0 / 32) * ka.stride_sfb));
    const auto r0 = rsrc(pa, 0u);
    // A DMA: lane -> row lane/8 of the 8-row block, LDS slot lane%8 holds chunk (lane%8) ^ swz
    const int dma_r = lane / 8, dma_c = (lane % 8) ^ ((dma_r >> 1) & 3);
    const int dma_goff = dma_r * ka.stride_a + dma_c * 16;
    auto rd_a = [&](int e, int kk) {   // A operand: row R = wrow + 16e + lane%16, chunk C = 4kk + lane/16
        const int R = wrow + 16 * e + lane % 16, C = 4 * kk + lane / 16;
        return (R / 8) * BLK + (R % 8) * 128 + ((C ^ (((R % 8) >> 1) & 3)) * 16);
    };

    unsigned vsa[D][SA], vsb[D][SB];
    auto issue = [&](auto st_, int kt) {   // all copies of K tile kt into stage / scale slot st
        constexpr int st = decltype(st_)::value;
        const bool in = kt < kend;
        const auto ra = in ? ra_ : r0;  const auto rb = in ? rb_ : r0;
        const auto rsa = in ? rsa_ : r0; const auto rsb = in ? rsb_ : r0;
        char* base = lds_stage<st, STAGE>();
        static_for<A_DMA>([&](auto i) {
            const int b = wave * A_DMA + i();
#if defined(__HIP_DEVICE_COMPILE__)
            __builtin_amdgcn_raw_ptr_buffer_load_lds(ra, (__attribute__((address_space(3))) void*)(base + b * BLK),
                                                     16, dma_goff, b * 8 * ka.stride_a + kt * 128, 0, 0);
#else
            (void)ra; (void)b; (void)base;
#endif
        });
        static_for<B_DMA>([&](auto j) {
            const int g = wave * B_DMA + j(), jb = g / 2, kk = g % 2;
#if defined(__HIP_DEVICE_COMPILE__)
            __builtin_amdgcn_raw_ptr_buffer_load_lds(rb, (__attribute__((address_space(3))) void*)(base + A_BYTES + jb * 2048 + kk * 1024),
                                                     16, lane * 16, jb * 16 * Kb + kt * 2048 + kk * 1024, 0, 0);
#else
            (void)rb; (void)jb; (void)kk;
#endif
        });
        static_for<SA>([&](auto j) { vsa[st][j()] = __builtin_amdgcn_raw_buffer_load_b32(rsa, lane * 4, j() * ka.stride_sfa + kt * 256, 0); });
        static_for<SB>([&](auto j) { vsb[st][j()] = __builtin_amdgcn_raw_buffer_load_b32(rsb, lane * 4, j() * ka.stride_sfb + kt * 256, 0); });
    };

    f32x4 acc[EM][EN];
    static_for<EM>([&](auto e) { static_for<EN>([&](auto n) { acc[e()][n()] = f32x4{0.f, 0.f, 0.f, 0.f}; }); });
    const int nk = kend > kbeg ? kend - kbeg : 0;
    static_for<D - 1>([&](auto t) { issue(t, kbeg + t()); __builtin_amdgcn_sched_barrier(0); });
#pragma clang loop unroll(disable)
    for (int i0 = 0; i0 < nk; i0 += D) {
        static_for<D>([&](auto u) {
            constexpr int s = u(), sp = (s + D - 1) % D;
            const int kt = kbeg + i0 + u();
            asm volatile("s_waitcnt vmcnt(%0)" :: "n"((D - 2) * G) : "memory");   // K tile kt's copies (own) landed
            __builtin_amdgcn_s_barrier();                                         // ... and every wave's, and kt-1 is consumed
            __builtin_amdgcn_sched_barrier(0);
            // Read this K tile's operands BEFORE issuing the next copies: the compiler cannot tell an LDS-DMA into
            // another stage from these reads and would otherwise wait for the copies it just issued (vmcnt) first.
            const char* sb = lds_stage<s, STAGE>();
            i32x4 va[2][EM], vb[2][EN];
            static_for<2>([&](auto kk) {
                static_for<EM>([&](auto e) { va[kk()][e()] = *(const i32x4*)(sb + rd_a(e(), kk())); });
                static_for<EN>([&](auto n) { vb[kk()][n()] = *(const i32x4*)(sb + A_BYTES + (wn * EN + n()) * 2048 + kk() * 1024 + lane * 16); });
            });
            __builtin_amdgcn_sched_barrier(0);
            issue(ic<sp>{}, kt + D - 1);
            __builtin_amdgcn_sched_barrier(0);
            static_for<2>([&](auto kk) { static_for<EM>([&](auto e) { static_for<EN>([&](auto n) {
                const i32x8 a8 = __builtin_shufflevector(va[kk()][e()], va[kk()][e()], 0, 1, 2, 3, -1, -1, -1, -1);
                const i32x8 b8 = __builtin_shufflevector(vb[kk()][n()], vb[kk()][n()], 0, 1, 2, 3, -1, -1, -1, -1);
                constexpr int ja = e() / 2, opa = 2 * kk() + e() % 2, jb = n() / 2, opb = 2 * kk() + n() % 2;
                acc[e()][n()] = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(b8, a8, acc[e()][n()], 4, 4,
                                                                                 opb, (int)vsb[s][jb], opa, (int)vsa[s][ja]);
            }); }); });
            __builtin_amdgcn_sched_barrier(0);
        });
    }
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    if (S > 1 && !splitk_reduce<EM, EN, BM, BN>(acc, ka, true, lane, row0, wrow, ncol0, bz, sk, S)) return;
    store_c<EM, EN>(acc, ka, lane, row0, wrow, ncol0, bz);
}
// ---------------------------------------------------------------------------------------------------------------------
// B_LDS: A and B global -> LDS by LDS-DMA through a D-deep ring of LDS stages (one barrier per K tile), scales straight
// into registers.  Stage image: A = BM/8 blocks of 8 rows (one DMA = 64 lanes x 16 B), block stride 1088 B, row r of a
// block at r*128 with its 16 B chunk c at slot c ^ ((r >> 1) & 3) (the 16 lanes of an A ds_read_b128 hit distinct bank
// groups); B = the shuffle_weight image copied verbatim: 16-column block jb, K half kk = 1 KB at jb*2048 + kk*1024, read
// lane*16 (contiguous, conflict free).  The A / B DMAs of a K tile are spread round-robin over the 4 waves.
// Ring: K tile kt lives in stage (kt - kbeg) % D; iteration kt waits for its own copies (counted vmcnt: exactly D-2 younger
// copy groups are in flight), passes the barrier, refills the stage of K tile kt-1 with K tile kt+D-1, then computes kt.
// LAYOUT == ROWMAJOR: B is copied into LDS with the same 8-row-block / XOR-swizzled image as A (m7/m8 style) and read
// like A; scales are loaded per 16-row block (8 B) and packed with one v_perm, as in body_direct.
template<int BM, int BN, int D, int LAYOUT>
__device__ __forceinline__ void body_lds_rm(const kargs_t& ka) {
    constexpr bool RM = LAYOUT == OPUS_MXFP4_SMALL_ROWMAJOR;
    static_assert(BM == 32 || BM == 64 || BM == 128 || (RM && BM == 16));
    static_assert(BN == 128 || BN == 256 || (RM && BN == 64));
    constexpr int WM = (BM == 128) ? 2 : 1, WN = 4 / WM;           // 1x4 for BM <= 64, 2x2 for BM == 128
    static_assert(!(BM == 128 && BN != 128), "BM = 128 is 128 x 128 (2x2 waves)");
    constexpr int WROWS = BM / WM, WCOLS = BN / WN;
    constexpr int EM = WROWS / 16, EN = WCOLS / 16;
    static_assert(RM || (EM % 2 == 0 && EN % 2 == 0), "waves cover whole 32-row scale blocks");
    constexpr int SA = RM ? EM : WROWS / 32, SB = RM ? EN : WCOLS / 32;
    constexpr int BLK = 1088;
    constexpr int A_BYTES = BM / 8 * BLK, B_BYTES = RM ? BN / 8 * BLK : BN * 128;
    // RM: per-K-tile scale image in the stage, [row][8 B] for max(BM, 32) A rows then max(BN, 32) B rows, filled by 4-byte
    // LDS-DMA (one instruction = 32 rows x 8 B, block-wide) instead of per-wave 8-byte loads per 16-row block.
    constexpr int SFA_ROWS = BM < 32 ? 32 : BM, SFB_ROWS = BN < 32 ? 32 : BN;
    constexpr int SF_BYTES = RM ? (SFA_ROWS + SFB_ROWS) * 8 : 0;
    constexpr int STAGE = A_BYTES + B_BYTES + SF_BYTES;
    static_assert(D * STAGE <= 160 * 1024, "LDS ring too deep for this tile");
    static_assert(RM || (BM / 8 % 4 == 0 && BN / 16 * 2 % 4 == 0), "shuffled: DMAs split evenly over the 4 waves");
    // DMA instructions of a K tile, spread over the 4 waves; when a count is not a multiple of 4 the extra waves repeat the
    // last copy (same data to the same LDS bytes), so every wave issues the same number of VMEM instructions.
    constexpr int A_BLK = BM / 8, B_BLK = RM ? BN / 8 : BN / 16 * 2, SF_BLK = RM ? (SFA_ROWS + SFB_ROWS) / 32 : 0;
    constexpr int A_DMA = (A_BLK + 3) / 4, B_DMA = (B_BLK + 3) / 4, SF_DMA = (SF_BLK + 3) / 4;   // per wave per K tile
    constexpr int G = A_DMA + B_DMA + (RM ? SF_DMA : SA + SB);         // VMEM instructions per wave per K tile
    static_assert((D - 2) * G <= 63, "vmcnt range");

    const int lane = __lane_id();
    const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x / 64);
    const int wm = wave / WN, wn = wave % WN;
    const int S = ka.splitk;
    const int col0 = blockIdx.x * BN, row0 = blockIdx.y * BM, bz = blockIdx.z / S, sk = blockIdx.z % S;
    const int ktot = ka.k / 256, kper = (ktot + S - 1) / S, kbeg = sk * kper;
    const int kend = (ktot < kbeg + kper ? ktot : kbeg + kper);
    const int Kb = ka.k / 2;
    const int wrow = wm * WROWS, ncol0 = col0 + wn * WCOLS;

    const uint8_t* pa  = (const uint8_t*)ka.ptr_a   + bz * ka.stride_a_batch   + (long long)row0 * ka.stride_a;
    const uint8_t* pb  = (const uint8_t*)ka.ptr_b   + bz * ka.stride_b_batch   + (long long)col0 * Kb;
    const uint8_t* psa = (const uint8_t*)ka.ptr_sfa + bz * ka.stride_sfa_batch + (long long)(RM ? row0 + wrow : (row0 + wrow) / 32) * ka.stride_sfa;
    const uint8_t* psb = (const uint8_t*)ka.ptr_sfb + bz * ka.stride_sfb_batch + (long long)(RM ? ncol0 : ncol0 / 32) * ka.stride_sfb;
    const auto ra_  = rsrc(pa,  clampb((long long)(ka.m - row0) * ka.stride_a));
    const auto rb_  = rsrc(pb,  clampb((long long)(ka.n - col0) * Kb));
    const auto rsa_ = rsrc(psa, clampb(RM ? (long long)(ka.m - row0 - wrow) * ka.stride_sfa : (long long)((ka.m + 31) / 32 - (row0 + wrow) / 32) * ka.stride_sfa));
    const auto rsb_ = rsrc(psb, clampb(RM ? (long long)(ka.n - ncol0) * ka.stride_sfb : (long long)((ka.n + 31) / 32 - ncol0 / 32) * ka.stride_sfb));
    const int sa_lane = RM ? (lane % 16) * ka.stride_sfa : lane * 4;
    const int sb_lane = RM ? (lane % 16) * ka.stride_sfb : lane * 4;
    const unsigned rsel = rm_sel(lane);
    const auto r0 = rsrc(pa, 0u);
    // RM: block-wide scale bases / sizes (rows from row0 / col0), used by the scale DMA.  Kept as readfirstlane'd scalars
    // and turned into a descriptor per copy: selecting between two descriptors made the compiler keep them in VGPRs
    // (a readfirstlane waterfall loop around every DMA).
    const uintptr_t sfa_base = (uintptr_t)((const uint8_t*)ka.ptr_sfa + bz * ka.stride_sfa_batch + (long long)row0 * ka.stride_sfa);
    const uintptr_t sfb_base = (uintptr_t)((const uint8_t*)ka.ptr_sfb + bz * ka.stride_sfb_batch + (long long)col0 * ka.stride_sfb);
    const unsigned sfa_lo = __builtin_amdgcn_readfirstlane((unsigned)sfa_base), sfa_hi = __builtin_amdgcn_readfirstlane((unsigned)(sfa_base >> 32));
    const unsigned sfb_lo = __builtin_amdgcn_readfirstlane((unsigned)sfb_base), sfb_hi = __builtin_amdgcn_readfirstlane((unsigned)(sfb_base >> 32));
    const unsigned sfa_sz = __builtin_amdgcn_readfirstlane(clampb((long long)(ka.m - row0) * ka.stride_sfa));
    const unsigned sfb_sz = __builtin_amdgcn_readfirstlane(clampb((long long)(ka.n - col0) * ka.stride_sfb));
    // A DMA: lane -> row lane/8 of the 8-row block, LDS slot lane%8 holds chunk (lane%8) ^ swz
    const int dma_r = lane / 8, dma_c = (lane % 8) ^ ((dma_r >> 1) & 3);
    const int dma_goff = dma_r * ka.stride_a + dma_c * 16;
    const int dma_goff_b = dma_r * Kb + dma_c * 16;               // RM: B DMA, same image as A
    auto rd_rm = [&](int R0, int kk) {   // row-major image: row R = R0 + lane%16, chunk C = 4kk + lane/16
        const int R = R0 + lane % 16, C = 4 * kk + lane / 16;
        return (R / 8) * BLK + (R % 8) * 128 + ((C ^ (((R % 8) >> 1) & 3)) * 16);
    };
    auto rd_a = [&](int e, int kk) { return rd_rm(wrow + 16 * e, kk); };

    unsigned vsa[D][RM ? 1 : SA], vsb[D][RM ? 1 : SB];
    // RM scale DMA: lane i of instruction q copies row 32q + i/2, bytes [4*(i%2), +4) of the K tile's 8 scale bytes.
    const int sfd_row = lane / 2, sfd_half = (lane % 2) * 4;
    auto issue = [&](auto st_, int kt) {   // all copies of K tile kt into stage / scale slot st
        constexpr int st = decltype(st_)::value;
        const bool in = kt < kend;
        const auto ra = in ? ra_ : r0;  const auto rb = in ? rb_ : r0;
        const auto rsa = in ? rsa_ : r0; const auto rsb = in ? rsb_ : r0;
        char* base = lds_stage<st, STAGE>();
        static_for<A_DMA>([&](auto i) {
            const int b0 = wave * A_DMA + i();
            const int b = RM ? __builtin_amdgcn_readfirstlane(b0 < A_BLK ? b0 : A_BLK - 1) : b0;   // RM may repeat a copy; keep it uniform (M0)
#if defined(__HIP_DEVICE_COMPILE__)
            __builtin_amdgcn_raw_ptr_buffer_load_lds(ra, (__attribute__((address_space(3))) void*)(base + b * BLK),
                                                     16, dma_goff, b * 8 * ka.stride_a + kt * 128, 0, 0);
#else
            (void)ra; (void)b; (void)base;
#endif
        });
        static_for<B_DMA>([&](auto j) {
            const int g0 = wave * B_DMA + j();
            const int g = RM ? __builtin_amdgcn_readfirstlane(g0 < B_BLK ? g0 : B_BLK - 1) : g0;
#if defined(__HIP_DEVICE_COMPILE__)
            if constexpr (RM)   // 8-row block g of B
                __builtin_amdgcn_raw_ptr_buffer_load_lds(rb, (__attribute__((address_space(3))) void*)(base + A_BYTES + g * BLK),
                                                         16, dma_goff_b, g * 8 * Kb + kt * 128, 0, 0);
            else {              // shuffle_weight: (16-row block g/2, K half g%2) = contiguous 1 KB
                const int jb = g / 2, kk = g % 2;
                __builtin_amdgcn_raw_ptr_buffer_load_lds(rb, (__attribute__((address_space(3))) void*)(base + A_BYTES + jb * 2048 + kk * 1024),
                                                         16, lane * 16, jb * 16 * Kb + kt * 2048 + kk * 1024, 0, 0);
            }
#else
            (void)rb; (void)g;
#endif
        });
        if constexpr (RM) {
            static_for<SF_DMA>([&](auto j) {
                const int q0 = wave * SF_DMA + j(), q = __builtin_amdgcn_readfirstlane(q0 < SF_BLK ? q0 : SF_BLK - 1);
                const bool isa = q < SFA_ROWS / 32;                    // block-uniform
                const unsigned lo = isa ? sfa_lo : sfb_lo, hi = isa ? sfa_hi : sfb_hi, sz = in ? (isa ? sfa_sz : sfb_sz) : 0u;
                const auto rsx = rsrc((const void*)(((uintptr_t)hi << 32) | lo), sz);
                const int stride = isa ? ka.stride_sfa : ka.stride_sfb, r = (isa ? q : q - SFA_ROWS / 32) * 32 + sfd_row;
#if defined(__HIP_DEVICE_COMPILE__)
                __builtin_amdgcn_raw_ptr_buffer_load_lds(rsx, (__attribute__((address_space(3))) void*)(base + A_BYTES + B_BYTES + q * 256),
                                                         4, r * stride + sfd_half, kt * 8, 0, 0);
#else
                (void)rsx; (void)r; (void)stride;
#endif
            });
        } else {
            static_for<SA>([&](auto j) { vsa[st][j()] = __builtin_amdgcn_raw_buffer_load_b32(rsa, sa_lane, j() * ka.stride_sfa + kt * 256, 0); });
            static_for<SB>([&](auto j) { vsb[st][j()] = __builtin_amdgcn_raw_buffer_load_b32(rsb, sb_lane, j() * ka.stride_sfb + kt * 256, 0); });
        }
    };

    f32x4 acc[EM][EN];
    static_for<EM>([&](auto e) { static_for<EN>([&](auto n) { acc[e()][n()] = f32x4{0.f, 0.f, 0.f, 0.f}; }); });
    const int nk = kend > kbeg ? kend - kbeg : 0;
#ifndef SMALL_RM_PIPE2
#define SMALL_RM_PIPE2 0   // asm-style row-major B_LDS schedule: mixed results (see EXPERIMENTS.md), off by default
#endif
    if constexpr (RM && SMALL_RM_PIPE2) {
        // asm-style: operands (A, B, scales) of tile kt+1 are read from LDS while the MFMAs of kt run, and the copies of
        // kt+D (into kt's freed stage) are interleaved with those MFMAs.  Iteration kt waits until group kt+1 landed.
        static_assert((D - 1) * G <= 63, "vmcnt range");
        static_for<D>([&](auto t) { issue(t, kbeg + t()); __builtin_amdgcn_sched_barrier(0); });
        i32x4 va[2][EM], vb[2][EN]; u32x2 rsa_raw[EM], rsb_raw[EN];
        auto rd_all = [&](auto st_, i32x4 (&xa)[2][EM], i32x4 (&xb)[2][EN], u32x2 (&xsa)[EM], u32x2 (&xsb)[EN]) {
            constexpr int st = decltype(st_)::value;
            const char* sb = lds_stage<st, STAGE>();
            static_for<2>([&](auto kk) {
                static_for<EM>([&](auto e) { xa[kk()][e()] = *(const i32x4*)(sb + rd_a(e(), kk())); });
                static_for<EN>([&](auto n) { xb[kk()][n()] = *(const i32x4*)(sb + A_BYTES + rd_rm(16 * (wn * EN + n()), kk())); });
            });
            const char* sf = sb + A_BYTES + B_BYTES;
            static_for<EM>([&](auto e) { xsa[e()] = *(const u32x2*)(sf + (wrow + 16 * e() + lane % 16) * 8); });
            static_for<EN>([&](auto n) { xsb[n()] = *(const u32x2*)(sf + (SFA_ROWS + wn * WCOLS + 16 * n() + lane % 16) * 8); });
        };
        asm volatile("s_waitcnt vmcnt(%0)" :: "n"((D - 1) * G) : "memory");
        __builtin_amdgcn_s_barrier();
        rd_all(ic<0>{}, va, vb, rsa_raw, rsb_raw);
        __builtin_amdgcn_sched_barrier(0);
        constexpr int NR = 2 * EM + 2 * EN + EM + EN;   // ds_reads per tile
#pragma clang loop unroll(disable)
        for (int i0 = 0; i0 < nk; i0 += D) {
            static_for<D>([&](auto u) {
                constexpr int s = u(), sn = (s + 1) % D;
                const int kt = kbeg + i0 + u();
                asm volatile("s_waitcnt vmcnt(%0) lgkmcnt(0)" :: "n"((D - 2) * G) : "memory");
                __builtin_amdgcn_s_barrier();
                __builtin_amdgcn_sched_barrier(0);
                i32x4 na[2][EM], nb[2][EN]; u32x2 nsa[EM], nsb[EN];
                rd_all(ic<sn>{}, na, nb, nsa, nsb);
                unsigned pa_[EM], pb_[EN];
                static_for<EM>([&](auto e) { pa_[e()] = rm_pack(rsa_raw[e()], rsel); });
                static_for<EN>([&](auto n) { pb_[n()] = rm_pack(rsb_raw[n()], rsel); });
                static_for<2>([&](auto kk) { static_for<EM>([&](auto e) { static_for<EN>([&](auto n) {
                    const i32x8 a8 = __builtin_shufflevector(va[kk()][e()], va[kk()][e()], 0, 1, 2, 3, -1, -1, -1, -1);
                    const i32x8 b8 = __builtin_shufflevector(vb[kk()][n()], vb[kk()][n()], 0, 1, 2, 3, -1, -1, -1, -1);
                    acc[e()][n()] = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(b8, a8, acc[e()][n()], 4, 4,
                                                                                     kk(), (int)pb_[n()], kk(), (int)pa_[e()]);
                }); }); });
                issue(ic<s>{}, kt + D);
                __builtin_amdgcn_sched_group_barrier(0x100, NR, 0);
                static_for<2 * EM * EN>([&](auto m) {
                    __builtin_amdgcn_sched_group_barrier(0x008, 1, 0);
                    if constexpr (m() < G) __builtin_amdgcn_sched_group_barrier(0x020 | 0x800, 1, 0);
                });
                if constexpr (2 * EM * EN < G) __builtin_amdgcn_sched_group_barrier(0x020 | 0x800, G - 2 * EM * EN, 0);
                __builtin_amdgcn_sched_barrier(0);
                static_for<2>([&](auto kk) {
                    static_for<EM>([&](auto e) { va[kk()][e()] = na[kk()][e()]; });
                    static_for<EN>([&](auto n) { vb[kk()][n()] = nb[kk()][n()]; });
                });
                static_for<EM>([&](auto e) { rsa_raw[e()] = nsa[e()]; });
                static_for<EN>([&](auto n) { rsb_raw[n()] = nsb[n()]; });
            });
        }
    } else {
    const int nk = kend > kbeg ? kend - kbeg : 0;
        static_for<D - 1>([&](auto t) { issue(t, kbeg + t()); __builtin_amdgcn_sched_barrier(0); });
    #pragma clang loop unroll(disable)
        for (int i0 = 0; i0 < nk; i0 += D) {
            static_for<D>([&](auto u) {
                constexpr int s = u(), sp = (s + D - 1) % D;
                const int kt = kbeg + i0 + u();
                asm volatile("s_waitcnt vmcnt(%0)" :: "n"((D - 2) * G) : "memory");   // K tile kt's copies (own) landed
                __builtin_amdgcn_s_barrier();                                         // ... and every wave's, and kt-1 is consumed
                __builtin_amdgcn_sched_barrier(0);
                // Read this K tile's operands BEFORE issuing the next copies: the compiler cannot tell an LDS-DMA into
                // another stage from these reads and would otherwise wait for the copies it just issued (vmcnt) first.
                const char* sb = lds_stage<s, STAGE>();
                i32x4 va[2][EM], vb[2][EN];
                static_for<2>([&](auto kk) {
                    static_for<EM>([&](auto e) { va[kk()][e()] = *(const i32x4*)(sb + rd_a(e(), kk())); });
                    static_for<EN>([&](auto n) {
                        const int ob = RM ? rd_rm(16 * (wn * EN + n()), kk()) : (wn * EN + n()) * 2048 + kk() * 1024 + lane * 16;
                        vb[kk()][n()] = *(const i32x4*)(sb + A_BYTES + ob);
                    });
                });
                u32x2 rsa_raw[RM ? EM : 1], rsb_raw[RM ? EN : 1];
                if constexpr (RM) {
                    const char* sf = sb + A_BYTES + B_BYTES;
                    static_for<EM>([&](auto e) { rsa_raw[e()] = *(const u32x2*)(sf + (wrow + 16 * e() + lane % 16) * 8); });
                    static_for<EN>([&](auto n) { rsb_raw[n()] = *(const u32x2*)(sf + (SFA_ROWS + wn * WCOLS + 16 * n() + lane % 16) * 8); });
                }
                __builtin_amdgcn_sched_barrier(0);
                issue(ic<sp>{}, kt + D - 1);
                __builtin_amdgcn_sched_barrier(0);
                unsigned pa_[RM ? EM : 1], pb_[RM ? EN : 1];
                if constexpr (RM) {
                    static_for<EM>([&](auto e) { pa_[e()] = rm_pack(rsa_raw[e()], rsel); });
                    static_for<EN>([&](auto n) { pb_[n()] = rm_pack(rsb_raw[n()], rsel); });
                }
                static_for<2>([&](auto kk) { static_for<EM>([&](auto e) { static_for<EN>([&](auto n) {
                    const i32x8 a8 = __builtin_shufflevector(va[kk()][e()], va[kk()][e()], 0, 1, 2, 3, -1, -1, -1, -1);
                    const i32x8 b8 = __builtin_shufflevector(vb[kk()][n()], vb[kk()][n()], 0, 1, 2, 3, -1, -1, -1, -1);
                    if constexpr (RM) {
                        acc[e()][n()] = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(b8, a8, acc[e()][n()], 4, 4,
                                                                                         kk(), (int)pb_[n()], kk(), (int)pa_[e()]);
                    } else {
                        constexpr int ja = e() / 2, opa = 2 * kk() + e() % 2, jb = n() / 2, opb = 2 * kk() + n() % 2;
                        acc[e()][n()] = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(b8, a8, acc[e()][n()], 4, 4,
                                                                                         opb, (int)vsb[s][jb], opa, (int)vsa[s][ja]);
                    }
                }); }); });
                __builtin_amdgcn_sched_barrier(0);
            });
        }
    }
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    if (S > 1 && !splitk_reduce<EM, EN, BM, BN>(acc, ka, true, lane, row0, wrow, ncol0, bz, sk, S)) return;
    store_c<EM, EN>(acc, ka, lane, row0, wrow, ncol0, bz);
}
// ---------------------------------------------------------------------------------------------------------------------
// A_LDS (shuffled): A global -> LDS by LDS-DMA through a D-deep ring of LDS stages (shared by the 4 waves, one barrier per
// K tile); B (shuffle_weight, contiguous 1 KB per operand) and the scales go straight into a D-deep register ring.
// 1x4 waves: wave w owns all BM rows and columns [col0 + w*BN/4, +BN/4).  Same ring discipline as body_lds: iteration kt
// waits for its own group (exactly D-2 younger groups in flight), barrier, reads A of kt from LDS, refills the slot of kt-1
// with kt+D-1, computes kt.
template<int BM, int BN, int D>
__device__ __forceinline__ void body_alds(const kargs_t& ka) {
    static_assert(BM == 32 || BM == 64 || BM == 128);
    static_assert(BN == 64 || BN == 128 || BN == 256);
    constexpr int WCOLS = BN / 4, EM = BM / 16, EN = WCOLS / 16;
    static_assert(EN % 2 == 0 || EN == 1, "");
    constexpr int SA = BM / 32, SB = (EN + 1) / 2;
    constexpr int BLK = 1088, A_BYTES = BM / 8 * BLK;
    static_assert(D * A_BYTES <= 160 * 1024, "LDS ring too deep for this tile");
    constexpr int A_BLK = BM / 8, A_DMA = (A_BLK + 3) / 4;
    constexpr int G = A_DMA + SA + 2 * EN + SB;                       // VMEM instructions per wave per K tile
    static_assert((D - 1) * G <= 63, "vmcnt range: ring too deep for this tile");

    const int lane = __lane_id();
    const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x / 64);
    const int S = ka.splitk;
    const int col0 = blockIdx.x * BN, row0 = blockIdx.y * BM, bz = blockIdx.z / S, sk = blockIdx.z % S;
    const int ktot = ka.k / 256, kper = (ktot + S - 1) / S, kbeg = sk * kper;
    const int kend = (ktot < kbeg + kper ? ktot : kbeg + kper);
    const int Kb = ka.k / 2;
    const int ncol0 = col0 + wave * WCOLS;

    const uint8_t* pa  = (const uint8_t*)ka.ptr_a   + bz * ka.stride_a_batch   + (long long)row0 * ka.stride_a;
    const uint8_t* pb  = (const uint8_t*)ka.ptr_b   + bz * ka.stride_b_batch   + (long long)ncol0 * Kb;
    const uint8_t* psa = (const uint8_t*)ka.ptr_sfa + bz * ka.stride_sfa_batch + (long long)(row0 / 32) * ka.stride_sfa;
    const uint8_t* psb = (const uint8_t*)ka.ptr_sfb + bz * ka.stride_sfb_batch + (long long)(ncol0 / 32) * ka.stride_sfb;
    const auto ra_  = rsrc(pa,  clampb((long long)(ka.m - row0) * ka.stride_a));
    const auto rb_  = rsrc(pb,  clampb((long long)(ka.n - ncol0) * Kb));
    const auto rsa_ = rsrc(psa, clampb((long long)((ka.m + 31) / 32 - row0 / 32) * ka.stride_sfa));
    const auto rsb_ = rsrc(psb, clampb((long long)((ka.n + 31) / 32 - ncol0 / 32) * ka.stride_sfb));
    const auto r0 = rsrc(pa, 0u);
    const unsigned sh_b = (EN == 1) ? 8u * ((ncol0 / 16) & 1) : 0u;   // one 16-col block: half of the 32-row scale block
    const int dma_r = lane / 8, dma_c = (lane % 8) ^ ((dma_r >> 1) & 3);
    const int dma_goff = dma_r * ka.stride_a + dma_c * 16;
    auto rd_a = [&](int e, int kk) {
        const int R = 16 * e + lane % 16, C = 4 * kk + lane / 16;
        return (R / 8) * BLK + (R % 8) * 128 + ((C ^ (((R % 8) >> 1) & 3)) * 16);
    };

    i32x4 vb[D][EN][2];
    unsigned vsa[D][SA], vsb[D][SB];
    auto issue = [&](auto st_, int kt) {
        constexpr int st = decltype(st_)::value;
        const bool in = kt < kend;
        const auto ra = in ? ra_ : r0;  const auto rb = in ? rb_ : r0;
        const auto rsa = in ? rsa_ : r0; const auto rsb = in ? rsb_ : r0;
        char* base = lds_stage<st, A_BYTES>();
        static_for<A_DMA>([&](auto i) {
            const int b0 = wave * A_DMA + i();
            const int b = (A_BLK % 4 == 0) ? b0 : __builtin_amdgcn_readfirstlane(b0 < A_BLK ? b0 : A_BLK - 1);
#if defined(__HIP_DEVICE_COMPILE__)
            __builtin_amdgcn_raw_ptr_buffer_load_lds(ra, (__attribute__((address_space(3))) void*)(base + b * BLK),
                                                     16, dma_goff, b * 8 * ka.stride_a + kt * 128, 0, 0);
#else
            (void)ra; (void)b; (void)base;
#endif
        });
        static_for<SA>([&](auto j) { vsa[st][j()] = __builtin_amdgcn_raw_buffer_load_b32(rsa, lane * 4, j() * ka.stride_sfa + kt * 256, 0); });
        static_for<EN>([&](auto e) { static_for<2>([&](auto kk) {
            vb[st][e()][kk()] = __builtin_bit_cast(i32x4, __builtin_amdgcn_raw_buffer_load_b128(rb, lane * 16, e() * 16 * Kb + kt * 2048 + kk() * 1024, 0));
        }); });
        static_for<SB>([&](auto j) { vsb[st][j()] = __builtin_amdgcn_raw_buffer_load_b32(rsb, lane * 4, j() * ka.stride_sfb + kt * 256, 0); });
    };

    f32x4 acc[EM][EN];
    static_for<EM>([&](auto e) { static_for<EN>([&](auto n) { acc[e()][n()] = f32x4{0.f, 0.f, 0.f, 0.f}; }); });
    const int nk = kend > kbeg ? kend - kbeg : 0;
    // asm-style pipeline: K tile t lives in slot t % D (LDS stage for A, registers for B / scales).  A operands are read
    // from LDS one tile ahead: iteration kt waits until group kt+1 landed (D-2 younger groups in flight), barrier (A(kt+1)
    // visible; every wave finished reading A(kt), so its stage is free), ds_reads A(kt+1), and runs the MFMAs of kt
    // interleaved with the copies of tile kt+D into kt's slot (one VMEM per MFMA).
    static_for<D>([&](auto t) { issue(t, kbeg + t()); __builtin_amdgcn_sched_barrier(0); });
    i32x4 va[2][EM];   // A operands of the current tile
    asm volatile("s_waitcnt vmcnt(%0)" :: "n"((D - 1) * G) : "memory");
    __builtin_amdgcn_s_barrier();
    static_for<2>([&](auto kk) { static_for<EM>([&](auto e) { va[kk()][e()] = *(const i32x4*)(lds_stage<0, A_BYTES>() + rd_a(e(), kk())); }); });
    __builtin_amdgcn_sched_barrier(0);
    static_assert(D % 2 == 0 || true, "");
#pragma clang loop unroll(disable)
    for (int i0 = 0; i0 < nk; i0 += D) {
        static_for<D>([&](auto u) {
            constexpr int s = u(), sn = (s + 1) % D;
            const int kt = kbeg + i0 + u();
            asm volatile("s_waitcnt vmcnt(%0) lgkmcnt(0)" :: "n"((D - 2) * G) : "memory");   // group kt+1 landed; A(kt) reads done
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);
            i32x4 vn[2][EM];
            static_for<2>([&](auto kk) { static_for<EM>([&](auto e) { vn[kk()][e()] = *(const i32x4*)(lds_stage<sn, A_BYTES>() + rd_a(e(), kk())); }); });
            static_for<2>([&](auto kk) { static_for<EM>([&](auto e) { static_for<EN>([&](auto n) {
                const i32x8 a8 = __builtin_shufflevector(va[kk()][e()], va[kk()][e()], 0, 1, 2, 3, -1, -1, -1, -1);
                const i32x8 b8 = __builtin_shufflevector(vb[s][n()][kk()], vb[s][n()][kk()], 0, 1, 2, 3, -1, -1, -1, -1);
                constexpr int ja = e() / 2, opa = 2 * kk() + e() % 2, jb = n() / 2, opb = 2 * kk() + ((EN == 1) ? 0 : n() % 2);
                const unsigned sb = (EN == 1) ? (vsb[s][jb] >> sh_b) : vsb[s][jb];
                acc[e()][n()] = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(b8, a8, acc[e()][n()], 4, 4,
                                                                                 opb, (int)sb, opa, (int)vsa[s][ja]);
            }); }); });
            issue(ic<s>{}, kt + D);
            // schedule: the next tile's ds_reads first, then MFMA / VMEM alternating
            __builtin_amdgcn_sched_group_barrier(0x100, 2 * EM, 0);
            static_for<2 * EM * EN>([&](auto m) {
                __builtin_amdgcn_sched_group_barrier(0x008, 1, 0);
                if constexpr (m() < G) __builtin_amdgcn_sched_group_barrier(0x020 | 0x800, 1, 0);
            });
            if constexpr (2 * EM * EN < G) __builtin_amdgcn_sched_group_barrier(0x020 | 0x800, G - 2 * EM * EN, 0);
            __builtin_amdgcn_sched_barrier(0);
            static_for<2>([&](auto kk) { static_for<EM>([&](auto e) { va[kk()][e()] = vn[kk()][e()]; }); });
        });
    }
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    if (S > 1 && !splitk_reduce<EM, EN, BM, BN>(acc, ka, true, lane, row0, 0, ncol0, bz, sk, S)) return;
    store_c<EM, EN>(acc, ka, lane, row0, 0, ncol0, bz);
}

}   // namespace opus_mxfp4_small

// grid = ((N + BN - 1) / BN, (M + BM - 1) / BM, batch * splitk), block = 256.
template<class Traits>
__global__ __launch_bounds__(256, 1) void gemm_a4w4_mxfp4_nt_small_kernel(opus_gemm_mxfp4_small_kargs_gfx950 ka) {
#if defined(__gfx950__) || !defined(__HIP_DEVICE_COMPILE__)
    using T = Traits;
    if constexpr (T::LAYOUT == OPUS_MXFP4_SMALL_ROWMAJOR && T::BMODE != OPUS_MXFP4_SMALL_A_LDS) {
        if constexpr (T::BMODE == OPUS_MXFP4_SMALL_B_DIRECT) opus_mxfp4_small::body_direct_rm<T::BM, T::BN, T::D, T::KS, T::LAYOUT>(ka);
        else                                                 opus_mxfp4_small::body_lds_rm<T::BM, T::BN, T::D, T::LAYOUT>(ka);
    } else if constexpr (T::BMODE == OPUS_MXFP4_SMALL_A_LDS) {
        opus_mxfp4_small::body_alds<T::BM, T::BN, T::D>(ka);
    } else {
        if constexpr (T::BMODE == OPUS_MXFP4_SMALL_B_DIRECT) opus_mxfp4_small::body_direct<T::BM, T::BN, T::D, T::KS>(ka);
        else                                                 opus_mxfp4_small::body_lds<T::BM, T::BN, T::D>(ka);
    }
#endif
}
