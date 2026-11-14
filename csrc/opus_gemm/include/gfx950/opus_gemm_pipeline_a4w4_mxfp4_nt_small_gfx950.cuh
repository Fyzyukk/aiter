// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Small-tile MXFP4 x MXFP4 NT GEMM (gfx950), shuffle layout (aiter gemm_a4w4(bpreshuffle=True)): A [M, K/2] fp4x2
// row-major, B shuffle_weight(16, 16), scales e8m0_shuffle (1x32 UE8M0), C [M, N] bf16.  M <= ~512.  Standard ROCm
// toolchain (no pin attributes, no opus).  Row-major inputs: opus_gemm_pipeline_a4w4_mxfp4_nt_small_noshuffle_gfx950.cuh.
//
// One operand path, A_LDS (body_alds): A global -> LDS by LDS-DMA (D-deep ring, shared by the 4 waves), B + scales
//   global -> a D-deep register ring (asm 32x128 style); KS = 2 / 4 splits the 4 waves along K (partials summed through
//   LDS at the end).
// MFMA v_mfma_scale_f32_16x16x128_f8f6f4 with B as srcA (each lane holds 4 consecutive columns of one C row);
//   scales read as raw e8m0_shuffle dwords ([kk0 half0, kk0 half1, kk1 half0, kk1 half1] of a 32-row block) with a
//   compile-time opsel; split-K over blocks (last-arriving block reduces); M / N tails by buffer ranges + 16-column C mask.
#pragma once
#include "opus_gemm_traits_a4w4_mxfp4_nt_small_gfx950.cuh"
#include "opus_gemm_pipeline_a4w4_mxfp4_nt_small_common_gfx950.cuh"

namespace opus_mxfp4_small {

// A_LDS: A global -> LDS by LDS-DMA through a D-deep ring of LDS stages (shared by the 4 waves, one barrier per ring
// step); B (shuffle_weight, contiguous 1 KB per operand) and the scales go straight into a D-deep register ring.
// KS = 1: 1x4 waves, wave w owns all BM rows and columns [col0 + w*BN/4, +BN/4).
// KS > 1: the 4 waves split KS ways along K and 4/KS ways along N.  One ring step then covers KS consecutive K tiles:
//   the stage holds A of all KS tiles, wave group g computes tile step*KS + g; partials summed through LDS at the end.
// A stage image: BM/8 blocks of 8 rows (one DMA = 64 lanes x 16 B), block stride 1088 B, row r of a block at r*128 with
//   its 16 B chunk c at slot c ^ ((r >> 1) & 3) (the 16 lanes of an A ds_read_b128 hit distinct bank groups); one
//   __shared__ object per stage (lds_stage).  The ring schedule is described at the K loop below.
// BM == 16: one 16-row block, the A scale is half of a 32-row e8m0_shuffle block (byte shift).
template<int BM, int BN, int D, int KS>
__device__ __forceinline__ void body_alds(const kargs_t& ka) {
    static_assert(BM == 16 || BM == 32 || BM == 64 || BM == 128);
    static_assert(KS == 1 || KS == 2 || KS == 4);
    constexpr int WN = 4 / KS;
    static_assert(BN == 32 || BN == 64 || BN == 128 || BN == 256);
    constexpr int WCOLS = BN / WN, EM = BM / 16, EN = WCOLS / 16;
    static_assert(EN % 2 == 0 || EN == 1, "a wave covers one 16-column block or whole 32-row scale blocks");
    constexpr int SA = (BM + 31) / 32, SB = (EN + 1) / 2;
    constexpr int BLK = 1088, A_BYTES = BM / 8 * BLK, STAGE = KS * A_BYTES;
    static_assert(D * STAGE <= 160 * 1024, "LDS ring too deep for this tile");
    constexpr int A_BLK = KS * BM / 8, A_DMA = (A_BLK + 3) / 4;
    constexpr int G = A_DMA + SA + 2 * EN + SB;                       // VMEM instructions per wave per ring step
    static_assert((D - 1) * G <= 63, "vmcnt range: ring too deep for this tile");

    const int lane = __lane_id();
    const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x / 64);
    const int S = ka.splitk;
    const int col0 = blockIdx.x * BN, row0 = blockIdx.y * BM, bz = blockIdx.z / S, sk = blockIdx.z % S;
    const int ktot = ka.k / 256, kper = (ktot + S - 1) / S, kbeg = sk * kper;
    const int kend = (ktot < kbeg + kper ? ktot : kbeg + kper);
    const int Kb = ka.k / 2;
    const int wn = KS == 1 ? wave : wave % WN, kgrp = KS == 1 ? 0 : wave / WN;
    const int ncol0 = col0 + wn * WCOLS;

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
    const unsigned sh_a = (BM == 16) ? 8u * ((row0 / 16) & 1) : 0u;    // BM == 16: half of the 32-row scale block
    const int dma_r = lane / 8, dma_c = (lane % 8) ^ ((dma_r >> 1) & 3);
    const int dma_goff = dma_r * ka.stride_a + dma_c * 16;
    auto rd_a = [&](int e, int kk) {
        const int R = 16 * e + lane % 16, C = 4 * kk + lane / 16;
        return (KS == 1 ? 0 : kgrp * A_BYTES) + (R / 8) * BLK + (R % 8) * 128 + ((C ^ (((R % 8) >> 1) & 3)) * 16);
    };

    i32x4 vb[D][EN][2];
    unsigned vsa[D][SA], vsb[D][SB];
    auto issue = [&](auto st_, int kt) {   // KS > 1: A of tiles kt .. kt+KS-1, B / scales of tile kt + kgrp
        constexpr int st = decltype(st_)::value;
        const int kw = kt + kgrp;
        const bool in = kw < kend;
        const auto ra = in ? ra_ : r0;  const auto rb = in ? rb_ : r0;
        const auto rsa = in ? rsa_ : r0; const auto rsb = in ? rsb_ : r0;
        char* base = lds_stage<st, STAGE>();
        static_for<A_DMA>([&](auto i) {
            const int b0 = wave * A_DMA + i();
            const int b = (A_BLK % 4 == 0) ? b0 : __builtin_amdgcn_readfirstlane(b0 < A_BLK ? b0 : A_BLK - 1);
            const int t = KS == 1 ? 0 : b / (BM / 8), r = KS == 1 ? b : b % (BM / 8);   // K tile kt+t, 8-row block r
            const auto rat = KS == 1 ? ra : (kt + t < kend ? ra_ : r0);
#if defined(__HIP_DEVICE_COMPILE__)   // host pass: the LDS-DMA builtin is device-only
            __builtin_amdgcn_raw_ptr_buffer_load_lds(rat, (__attribute__((address_space(3))) void*)(base + b * BLK),
                                                     16, dma_goff, r * 8 * ka.stride_a + (kt + t) * 128, 0, 0);
#endif
        });
        static_for<SA>([&](auto j) { vsa[st][j()] = __builtin_amdgcn_raw_buffer_load_b32(rsa, lane * 4, j() * ka.stride_sfa + kw * 256, 0); });
        static_for<EN>([&](auto e) { static_for<2>([&](auto kk) {
            vb[st][e()][kk()] = __builtin_bit_cast(i32x4, __builtin_amdgcn_raw_buffer_load_b128(rb, lane * 16, e() * 16 * Kb + kw * 2048 + kk() * 1024, 0));
        }); });
        static_for<SB>([&](auto j) { vsb[st][j()] = __builtin_amdgcn_raw_buffer_load_b32(rsb, lane * 4, j() * ka.stride_sfb + kw * 256, 0); });
    };

    f32x4 acc[EM][EN];
    static_for<EM>([&](auto e) { static_for<EN>([&](auto n) { acc[e()][n()] = f32x4{0.f, 0.f, 0.f, 0.f}; }); });
    const int nk = kend > kbeg ? (kend - kbeg + KS - 1) / KS : 0;   // ring steps (KS K tiles each)
    // asm-style pipeline: ring step t lives in slot t % D (LDS stage for A, registers for B / scales).  A operands are read
    // from LDS one step ahead: step t waits until group t+1 landed (D-2 younger groups in flight), barrier (A(t+1)
    // visible; every wave finished reading A(t), so its stage is free), ds_reads A(t+1), and runs the MFMAs of t
    // interleaved with the copies of step t+D into t's slot (one VMEM per MFMA).
    static_for<D>([&](auto t) { issue(t, kbeg + KS * t()); __builtin_amdgcn_sched_barrier(0); });
    i32x4 va[2][EM];   // A operands of the current step
    asm volatile("s_waitcnt vmcnt(%0)" :: "n"((D - 1) * G) : "memory");
    __builtin_amdgcn_s_barrier();
    static_for<2>([&](auto kk) { static_for<EM>([&](auto e) { va[kk()][e()] = *(const i32x4*)(lds_stage<0, STAGE>() + rd_a(e(), kk())); }); });
    __builtin_amdgcn_sched_barrier(0);
#pragma clang loop unroll(disable)
    for (int i0 = 0; i0 < nk; i0 += D) {
        static_for<D>([&](auto u) {
            constexpr int s = u(), sn = (s + 1) % D;
            const int kt = kbeg + KS * (i0 + u());
            asm volatile("s_waitcnt vmcnt(%0) lgkmcnt(0)" :: "n"((D - 2) * G) : "memory");   // group t+1 landed; A(t) reads done
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);
            i32x4 vn[2][EM];
            static_for<2>([&](auto kk) { static_for<EM>([&](auto e) { vn[kk()][e()] = *(const i32x4*)(lds_stage<sn, STAGE>() + rd_a(e(), kk())); }); });
            static_for<2>([&](auto kk) { static_for<EM>([&](auto e) { static_for<EN>([&](auto n) {
                const i32x8 a8 = __builtin_shufflevector(va[kk()][e()], va[kk()][e()], 0, 1, 2, 3, -1, -1, -1, -1);
                const i32x8 b8 = __builtin_shufflevector(vb[s][n()][kk()], vb[s][n()][kk()], 0, 1, 2, 3, -1, -1, -1, -1);
                constexpr int ja = e() / 2, opa = 2 * kk() + ((BM == 16) ? 0 : e() % 2), jb = n() / 2, opb = 2 * kk() + ((EN == 1) ? 0 : n() % 2);
                const unsigned sb = (EN == 1) ? (vsb[s][jb] >> sh_b) : vsb[s][jb];
                const unsigned sa = (BM == 16) ? (vsa[s][ja] >> sh_a) : vsa[s][ja];
                acc[e()][n()] = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(b8, a8, acc[e()][n()], 4, 4,
                                                                                 opb, (int)sb, opa, (int)sa);
            }); }); });
            issue(ic<s>{}, kt + KS * D);
            // schedule: the next step's ds_reads first, then MFMA / VMEM alternating
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

}   // namespace opus_mxfp4_small

// grid = ((N + BN - 1) / BN, (M + BM - 1) / BM, batch * splitk), block = 256.
template<class Traits>
__global__ __launch_bounds__(256, 1) void gemm_a4w4_mxfp4_nt_small_kernel(opus_gemm_mxfp4_small_kargs_gfx950 ka) {
#if defined(__gfx950__)
    using T = Traits;
    opus_mxfp4_small::body_alds<T::BM, T::BN, T::D, T::KS>(ka);
#endif   // __gfx950__
}
