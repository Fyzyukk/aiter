// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Small-tile MXFP4 x MXFP4 NT GEMM (gfx950), row-major layout (aiter Triton gemm_afp4wfp4): A [M, K/2], B [N, K/2]
// fp4x2, scales [rows, K/32] UE8M0, all row-major, no padding; C [M, N] bf16.  N % 8 == 0, K % 256 == 0, any M.
// Standard ROCm toolchain (no pin attributes, no opus).  Shuffle inputs: opus_gemm_pipeline_a4w4_mxfp4_nt_small_gfx950.cuh.
//
// Two operand paths (traits BMODE):
//   B_DIRECT (body_direct_rm): A / B / scales global -> a D-deep register ring, no LDS in the K loop; KS splits the 4
//     waves along K, each wave group taking a contiguous K chunk (even D: one 16-byte scale load covers two K tiles).
//   B_LDS    (body_lds_rm): A and B global -> LDS by LDS-DMA (8-row-block, XOR-swizzled image), scales by a block-wide
//     4-byte LDS-DMA, one barrier per K tile.
// Scales: per 16-row block one 8-byte row load (or LDS read) + one v_perm, so the MFMA takes them with opsel = kk.
#pragma once
#include "opus_gemm_traits_a4w4_mxfp4_nt_small_noshuffle_gfx950.cuh"
#include "opus_gemm_pipeline_a4w4_mxfp4_nt_small_common_gfx950.cuh"

namespace opus_mxfp4_small {

// Row-major scales: for a 16-row block, lane L needs row L%16, K groups g + 4kk (g = L/16, kk = 0/1) of the K tile:
// one 8-byte load of the row's 8 groups, then one v_perm moves byte g -> byte 0 (kk 0) and byte 4+g -> byte 1 (kk 1),
// so the MFMA picks them with opsel = kk.
__device__ __forceinline__ unsigned rm_sel(int lane) { const unsigned g = lane / 16; return g | ((4u + g) << 8) | 0x0c0c0000u; }
__device__ __forceinline__ unsigned rm_pack(u32x2 v, unsigned sel) { return __builtin_amdgcn_perm(v[1], v[0], sel); }

// ---------------------------------------------------------------------------------------------------------------------
// B_DIRECT: no LDS in the K loop.  The 4 waves split KS ways along K (wave group kg takes a contiguous chunk of the
// block's K tiles) and 4 / KS ways along N; the KS partial accumulators are summed through LDS at the end (one barrier).
template<int BM, int BN, int D, int KS>
__device__ __forceinline__ void body_direct_rm(const kargs_t& ka) {
    static_assert(BM == 16 || BM == 32 || BM == 64 || BM == 128);
    static_assert(BN == 32 || BN == 64 || BN == 128 || BN == 256);
    constexpr int EM = BM / 16;
    static_assert(KS == 1 || KS == 2 || KS == 4);
    constexpr int WN = 4 / KS;
    constexpr int EN = BN / (16 * WN);
    constexpr int SA = EM, SB = EN;   // scale loads per K tile: one per 16-row block
    const int lane = __lane_id();
    const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x / 64);
    const int S = ka.splitk;
    const int col0 = blockIdx.x * BN, row0 = blockIdx.y * BM, bz = blockIdx.z / S, sk = blockIdx.z % S;
    const int ktot = ka.k / 256, kper = (ktot + S - 1) / S, kbeg = sk * kper;
    const int kend = (ktot < kbeg + kper ? ktot : kbeg + kper);     // this block's K tiles: [kbeg, kend)
    const int Kb = ka.k / 2;
    const int wn = wave % WN, kgrp = wave / WN;
    const int ncol0 = col0 + wn * (BN / WN);

    const uint8_t* pa  = (const uint8_t*)ka.ptr_a   + bz * ka.stride_a_batch   + (long long)row0 * ka.stride_a;
    const uint8_t* pb  = (const uint8_t*)ka.ptr_b   + bz * ka.stride_b_batch   + (long long)ncol0 * Kb;
    const uint8_t* psa = (const uint8_t*)ka.ptr_sfa + bz * ka.stride_sfa_batch + (long long)row0 * ka.stride_sfa;
    const uint8_t* psb = (const uint8_t*)ka.ptr_sfb + bz * ka.stride_sfb_batch + (long long)ncol0 * ka.stride_sfb;
    const auto ra_  = rsrc(pa,  clampb((long long)(ka.m - row0) * ka.stride_a));
    const auto rb_  = rsrc(pb,  clampb((long long)(ka.n - ncol0) * Kb));
    const auto rsa_ = rsrc(psa, clampb((long long)(ka.m - row0) * ka.stride_sfa));
    const auto rsb_ = rsrc(psb, clampb((long long)(ka.n - ncol0) * ka.stride_sfb));
    const int a_lane = (lane % 16) * ka.stride_a + (lane / 16) * 16;   // A operand: row lane%16, K bytes [16*(lane/16), +16)
    const int b_lane = (lane % 16) * Kb + (lane / 16) * 16;            // B operand addressed like A
    const int sa_lane = (lane % 16) * ka.stride_sfa;
    const int sb_lane = (lane % 16) * ka.stride_sfb;
    const unsigned rsel = rm_sel(lane);

    // K tiles of this wave: a contiguous chunk per wave group, so with an even ring depth (PAIR) the scales of tiles
    // (kt, kt+1) -- 16 contiguous bytes per row -- come in with one 16-byte load at the odd stage (half the scale
    // instructions, each touching 16 rows).
    constexpr bool PAIR = D % 2 == 0;
    const int kchunk = (kend - kbeg + KS - 1) / KS;
    const int k0 = kbeg + kgrp * kchunk;
    const int kend_w = kend < k0 + kchunk ? kend : k0 + kchunk;
    const int nk = kend_w > k0 ? kend_w - k0 : 0;

    i32x4 va[D][EM][2], vb[D][EN][2];
    u32x2 vsa[D][SA], vsb[D][SB];
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
            const int ob = e() * 16 * Kb + kt * 128 + kk() * 64;
            vb[s][e()][kk()] = __builtin_bit_cast(i32x4, __builtin_amdgcn_raw_buffer_load_b128(rb, b_lane, ob, 0));
        }); });
        static_for<EM>([&](auto e) { static_for<2>([&](auto kk) {
            va[s][e()][kk()] = __builtin_bit_cast(i32x4, __builtin_amdgcn_raw_buffer_load_b128(ra, a_lane, e() * 16 * ka.stride_a + kt * 128 + kk() * 64, 0));
        }); });
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
    };

    f32x4 acc[EM][EN];
    static_for<EM>([&](auto e) { static_for<EN>([&](auto n) { acc[e()][n()] = f32x4{0.f, 0.f, 0.f, 0.f}; }); });
    auto compute = [&](auto s_) {
        constexpr int s = decltype(s_)::value;
        unsigned pa_[EM], pb_[EN];
        if constexpr (PAIR) {
            static_for<EM>([&](auto e) { const u32x4 v = psa2[s / 2][e()]; pa_[e()] = rm_pack(s % 2 ? u32x2{v[2], v[3]} : u32x2{v[0], v[1]}, rsel); });
            static_for<EN>([&](auto n) { const u32x4 v = psb2[s / 2][n()]; pb_[n()] = rm_pack(s % 2 ? u32x2{v[2], v[3]} : u32x2{v[0], v[1]}, rsel); });
        } else {
            static_for<EM>([&](auto e) { pa_[e()] = rm_pack(vsa[s][e()], rsel); });
            static_for<EN>([&](auto n) { pb_[n()] = rm_pack(vsb[s][n()], rsel); });
        }
        static_for<2>([&](auto kk) { static_for<EM>([&](auto e) { static_for<EN>([&](auto n) {
            const i32x8 a8 = __builtin_shufflevector(va[s][e()][kk()], va[s][e()][kk()], 0, 1, 2, 3, -1, -1, -1, -1);
            const i32x8 b8 = __builtin_shufflevector(vb[s][n()][kk()], vb[s][n()][kk()], 0, 1, 2, 3, -1, -1, -1, -1);
            acc[e()][n()] = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(b8, a8, acc[e()][n()], 4, 4,
                                                                             kk(), (int)pb_[n()], kk(), (int)pa_[e()]);
        }); }); });
    };
    static_for<D>([&](auto s) { issue(s, k0 + s()); __builtin_amdgcn_sched_barrier(0); });   // same order as the loop
#pragma clang loop unroll(disable)
    for (int i0 = 0; i0 < nk; i0 += D) {
        static_for<D>([&](auto s) {
            const int i = i0 + s();
            compute(s);                          // i >= nk: all operands are 0 (zero-range loads), adds nothing
            __builtin_amdgcn_sched_barrier(0);   // keep the ring order: stage s computes, then reloads
            issue(s, k0 + (i + D));
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
// B_LDS: A, B and scales global -> LDS by LDS-DMA through a D-deep ring of LDS stages (one barrier per K tile).
// Stage image: A and B = 8-row blocks (one DMA = 64 lanes x 16 B), block stride 1088 B, row r of a block at r*128 with
// its 16 B chunk c at slot c ^ ((r >> 1) & 3) (the 16 lanes of a ds_read_b128 hit distinct bank groups); then the K
// tile's scales, [row][8 B], filled by a block-wide 4-byte LDS-DMA and packed with one v_perm.  The DMAs of a K tile are
// spread round-robin over the 4 waves.  1x4 waves for BM <= 64, 2x2 for 128 x 128.
// Ring: K tile kt lives in stage (kt - kbeg) % D; iteration kt waits for its own copies (counted vmcnt: exactly D-2 younger
// copy groups are in flight), passes the barrier, refills the stage of K tile kt-1 with K tile kt+D-1, then computes kt.
template<int BM, int BN, int D>
__device__ __forceinline__ void body_lds_rm(const kargs_t& ka) {
    static_assert(BM == 16 || BM == 32 || BM == 64 || BM == 128);
    static_assert(BN == 64 || BN == 128 || BN == 256);
    constexpr int WM = (BM == 128) ? 2 : 1, WN = 4 / WM;           // 1x4 for BM <= 64, 2x2 for BM == 128
    static_assert(!(BM == 128 && BN != 128), "BM = 128 is 128 x 128 (2x2 waves)");
    constexpr int WROWS = BM / WM, WCOLS = BN / WN;
    constexpr int EM = WROWS / 16, EN = WCOLS / 16;
    constexpr int BLK = 1088;
    constexpr int A_BYTES = BM / 8 * BLK, B_BYTES = BN / 8 * BLK;
    // Per-K-tile scale image in the stage, [row][8 B] for max(BM, 32) A rows then max(BN, 32) B rows, filled by 4-byte
    // LDS-DMA (one instruction = 32 rows x 8 B, block-wide) instead of per-wave 8-byte loads per 16-row block.
    constexpr int SFA_ROWS = BM < 32 ? 32 : BM, SFB_ROWS = BN < 32 ? 32 : BN;
    constexpr int SF_BYTES = (SFA_ROWS + SFB_ROWS) * 8;
    constexpr int STAGE = A_BYTES + B_BYTES + SF_BYTES;
    static_assert(D * STAGE <= 160 * 1024, "LDS ring too deep for this tile");
    // DMA instructions of a K tile, spread over the 4 waves; when a count is not a multiple of 4 the extra waves repeat the
    // last copy (same data to the same LDS bytes), so every wave issues the same number of VMEM instructions.
    constexpr int A_BLK = BM / 8, B_BLK = BN / 8, SF_BLK = (SFA_ROWS + SFB_ROWS) / 32;
    constexpr int A_DMA = (A_BLK + 3) / 4, B_DMA = (B_BLK + 3) / 4, SF_DMA = (SF_BLK + 3) / 4;   // per wave per K tile
    constexpr int G = A_DMA + B_DMA + SF_DMA;                       // VMEM instructions per wave per K tile
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

    // psa / psb / rsa_ / rsb_ (per-wave scale descriptors) are unused -- the scales come in by the block-wide DMA below --
    // but removing them (or reordering these lines) changes the compiler's SGPR assignment; kept for unchanged codegen.
    const uint8_t* pa  = (const uint8_t*)ka.ptr_a   + bz * ka.stride_a_batch   + (long long)row0 * ka.stride_a;
    const uint8_t* pb  = (const uint8_t*)ka.ptr_b   + bz * ka.stride_b_batch   + (long long)col0 * Kb;
    const uint8_t* psa = (const uint8_t*)ka.ptr_sfa + bz * ka.stride_sfa_batch + (long long)(row0 + wrow) * ka.stride_sfa;
    const uint8_t* psb = (const uint8_t*)ka.ptr_sfb + bz * ka.stride_sfb_batch + (long long)ncol0 * ka.stride_sfb;
    const auto ra_  = rsrc(pa,  clampb((long long)(ka.m - row0) * ka.stride_a));
    const auto rb_  = rsrc(pb,  clampb((long long)(ka.n - col0) * Kb));
    [[maybe_unused]] const auto rsa_ = rsrc(psa, clampb((long long)(ka.m - row0 - wrow) * ka.stride_sfa));
    [[maybe_unused]] const auto rsb_ = rsrc(psb, clampb((long long)(ka.n - ncol0) * ka.stride_sfb));
    const unsigned rsel = rm_sel(lane);
    const auto r0 = rsrc(pa, 0u);
    // Block-wide scale bases / sizes (rows from row0 / col0), used by the scale DMA.  Kept as readfirstlane'd scalars
    // and turned into a descriptor per copy: selecting between two descriptors made the compiler keep them in VGPRs
    // (a readfirstlane waterfall loop around every DMA).
    const uintptr_t sfa_base = (uintptr_t)((const uint8_t*)ka.ptr_sfa + bz * ka.stride_sfa_batch + (long long)row0 * ka.stride_sfa);
    const uintptr_t sfb_base = (uintptr_t)((const uint8_t*)ka.ptr_sfb + bz * ka.stride_sfb_batch + (long long)col0 * ka.stride_sfb);
    const unsigned sfa_lo = __builtin_amdgcn_readfirstlane((unsigned)sfa_base), sfa_hi = __builtin_amdgcn_readfirstlane((unsigned)(sfa_base >> 32));
    const unsigned sfb_lo = __builtin_amdgcn_readfirstlane((unsigned)sfb_base), sfb_hi = __builtin_amdgcn_readfirstlane((unsigned)(sfb_base >> 32));
    const unsigned sfa_sz = __builtin_amdgcn_readfirstlane(clampb((long long)(ka.m - row0) * ka.stride_sfa));
    const unsigned sfb_sz = __builtin_amdgcn_readfirstlane(clampb((long long)(ka.n - col0) * ka.stride_sfb));
    // A / B DMA: lane -> row lane/8 of the 8-row block, LDS slot lane%8 holds chunk (lane%8) ^ swz
    const int dma_r = lane / 8, dma_c = (lane % 8) ^ ((dma_r >> 1) & 3);
    const int dma_goff = dma_r * ka.stride_a + dma_c * 16;
    const int dma_goff_b = dma_r * Kb + dma_c * 16;
    auto rd_rm = [&](int R0, int kk) {   // row R = R0 + lane%16, chunk C = 4kk + lane/16
        const int R = R0 + lane % 16, C = 4 * kk + lane / 16;
        return (R / 8) * BLK + (R % 8) * 128 + ((C ^ (((R % 8) >> 1) & 3)) * 16);
    };
    auto rd_a = [&](int e, int kk) { return rd_rm(wrow + 16 * e, kk); };

    // Scale DMA: lane i of instruction q copies row 32q + i/2, bytes [4*(i%2), +4) of the K tile's 8 scale bytes.
    const int sfd_row = lane / 2, sfd_half = (lane % 2) * 4;
    auto issue = [&](auto st_, int kt) {   // all copies of K tile kt into stage st
        constexpr int st = decltype(st_)::value;
        const bool in = kt < kend;
        const auto ra = in ? ra_ : r0;  const auto rb = in ? rb_ : r0;
        char* base = lds_stage<st, STAGE>();
        static_for<A_DMA>([&](auto i) {
            const int b0 = wave * A_DMA + i();
            const int b = __builtin_amdgcn_readfirstlane(b0 < A_BLK ? b0 : A_BLK - 1);   // may repeat a copy; keep it uniform (M0)
#if defined(__HIP_DEVICE_COMPILE__)   // host pass: the LDS-DMA builtin is device-only
            __builtin_amdgcn_raw_ptr_buffer_load_lds(ra, (__attribute__((address_space(3))) void*)(base + b * BLK),
                                                     16, dma_goff, b * 8 * ka.stride_a + kt * 128, 0, 0);
#endif
        });
        static_for<B_DMA>([&](auto j) {   // 8-row block g of B
            const int g0 = wave * B_DMA + j();
            const int g = __builtin_amdgcn_readfirstlane(g0 < B_BLK ? g0 : B_BLK - 1);
#if defined(__HIP_DEVICE_COMPILE__)   // host pass: the LDS-DMA builtin is device-only
            __builtin_amdgcn_raw_ptr_buffer_load_lds(rb, (__attribute__((address_space(3))) void*)(base + A_BYTES + g * BLK),
                                                     16, dma_goff_b, g * 8 * Kb + kt * 128, 0, 0);
#endif
        });
        static_for<SF_DMA>([&](auto j) {
            const int q0 = wave * SF_DMA + j(), q = __builtin_amdgcn_readfirstlane(q0 < SF_BLK ? q0 : SF_BLK - 1);
            const bool isa = q < SFA_ROWS / 32;                    // block-uniform
            const unsigned lo = isa ? sfa_lo : sfb_lo, hi = isa ? sfa_hi : sfb_hi, sz = in ? (isa ? sfa_sz : sfb_sz) : 0u;
            const auto rsx = rsrc((const void*)(((uintptr_t)hi << 32) | lo), sz);
            const int stride = isa ? ka.stride_sfa : ka.stride_sfb, r = (isa ? q : q - SFA_ROWS / 32) * 32 + sfd_row;
#if defined(__HIP_DEVICE_COMPILE__)   // host pass: the LDS-DMA builtin is device-only
            __builtin_amdgcn_raw_ptr_buffer_load_lds(rsx, (__attribute__((address_space(3))) void*)(base + A_BYTES + B_BYTES + q * 256),
                                                     4, r * stride + sfd_half, kt * 8, 0, 0);
#endif
        });
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
                static_for<EN>([&](auto n) { vb[kk()][n()] = *(const i32x4*)(sb + A_BYTES + rd_rm(16 * (wn * EN + n()), kk())); });
            });
            u32x2 rsa_raw[EM], rsb_raw[EN];
            const char* sf = sb + A_BYTES + B_BYTES;
            static_for<EM>([&](auto e) { rsa_raw[e()] = *(const u32x2*)(sf + (wrow + 16 * e() + lane % 16) * 8); });
            static_for<EN>([&](auto n) { rsb_raw[n()] = *(const u32x2*)(sf + (SFA_ROWS + wn * WCOLS + 16 * n() + lane % 16) * 8); });
            __builtin_amdgcn_sched_barrier(0);
            issue(ic<sp>{}, kt + D - 1);
            __builtin_amdgcn_sched_barrier(0);
            unsigned pa_[EM], pb_[EN];
            static_for<EM>([&](auto e) { pa_[e()] = rm_pack(rsa_raw[e()], rsel); });
            static_for<EN>([&](auto n) { pb_[n()] = rm_pack(rsb_raw[n()], rsel); });
            static_for<2>([&](auto kk) { static_for<EM>([&](auto e) { static_for<EN>([&](auto n) {
                const i32x8 a8 = __builtin_shufflevector(va[kk()][e()], va[kk()][e()], 0, 1, 2, 3, -1, -1, -1, -1);
                const i32x8 b8 = __builtin_shufflevector(vb[kk()][n()], vb[kk()][n()], 0, 1, 2, 3, -1, -1, -1, -1);
                acc[e()][n()] = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(b8, a8, acc[e()][n()], 4, 4,
                                                                                 kk(), (int)pb_[n()], kk(), (int)pa_[e()]);
            }); }); });
            __builtin_amdgcn_sched_barrier(0);
        });
    }
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    if (S > 1 && !splitk_reduce<EM, EN, BM, BN>(acc, ka, true, lane, row0, wrow, ncol0, bz, sk, S)) return;
    store_c<EM, EN>(acc, ka, lane, row0, wrow, ncol0, bz);
}

}   // namespace opus_mxfp4_small

// grid = ((N + BN - 1) / BN, (M + BM - 1) / BM, batch * splitk), block = 256.
template<class Traits>
__global__ __launch_bounds__(256, 1) void gemm_a4w4_mxfp4_nt_small_noshuffle_kernel(opus_gemm_mxfp4_small_kargs_gfx950 ka) {
#if defined(__gfx950__)
    using T = Traits;
    if constexpr (T::BMODE == OPUS_MXFP4_SMALL_B_DIRECT) opus_mxfp4_small::body_direct_rm<T::BM, T::BN, T::D, T::KS>(ka);
    else                                                 opus_mxfp4_small::body_lds_rm<T::BM, T::BN, T::D>(ka);
#endif   // __gfx950__
}
