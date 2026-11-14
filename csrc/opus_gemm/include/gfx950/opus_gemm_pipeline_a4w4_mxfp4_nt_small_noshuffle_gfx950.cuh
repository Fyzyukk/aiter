// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Small-tile MXFP4 x MXFP4 NT GEMM (gfx950), row-major layout (aiter Triton gemm_afp4wfp4): A [M, K/2], B [N, K/2]
// fp4x2, scales [rows, K/32] UE8M0, all row-major, no padding; C [M, N] bf16.  N % 8 == 0, K % 256 == 0, any M.
// opus, standard ROCm toolchain (no pin attributes).  Traits / kargs: opus_gemm_traits_a4w4_mxfp4_nt_small_noshuffle_gfx950.cuh.
// Same pipeline structure as the shuffle small-tile file (opus_gemm_pipeline_a4w4_mxfp4_nt_small_gfx950.cuh); independent
// of it (own namespace, no shared headers, both can be included in one TU).
//
// 1x4 waves (T_M = 1, T_N = 4): the 4 waves split the tile along N and share A.
//   A, B and the scales: global -> LDS by LDS-DMA into a STAGES-deep ring (one __shared__ object per stage); A / B as
//   8-row x 128 B chunks with an XOR swizzle (conflict-free ds_read_b128), the scales as a [row][8 B] image per K tile by
//   a block-wide 4-byte DMA.  Every wave ds_reads its operands one step ahead.  (Row-major rows are not MFMA-operand
//   shaped: loaded straight into registers, 16 rows x 64 B per instruction, B and the 8-byte scale rows were address-bound
//   at M >= 64.)  KS = 2 / 4 splits the 4 waves KS ways along K instead (T_N = 4 / KS; a ring step = KS K tiles); the
//   partial tiles are summed through LDS once.
// MFMA v_mfma_scale_f32_16x16x128_f8f6f4 with B as srcA (each lane holds 4 consecutive columns of one C row).  Scales:
//   per 16-row block one ds_read_b64 of row lane % 16 and one v_perm that puts K groups lane / 16 (kk 0) and 4 + lane / 16
//   (kk 1) into bytes 0 / 1, picked by the compile-time opsel = kk.
// Main loop (asm style): step t waits until step t+1 landed (counted vmcnt), barrier, ds_reads step t+1, and runs the
//   MFMAs of t interleaved with the copies of step t+STAGES into t's slot (sched_group_barrier).  K tiles past the end load
//   through zero-range buffers (no branch in the loop).
// Tails: M / N by buffer ranges (A rows, B rows, scale rows) + a 4-column C mask; no host padding.
// Split-K (split_k = S > 1): each block writes its fp32 partial tile to the workspace, splitk_reduce_kernel
//   (splitk_reduce_gfx950.cuh) sums the S partials into C.
#pragma once
#include "opus_gemm_traits_a4w4_mxfp4_nt_small_noshuffle_gfx950.cuh"

namespace opus_mxfp4_small_noshuffle {
using opus::operator""_I;
using fp4x32_t = opus::vector_t<opus::i32_t, 4>;   // one fp4 MFMA operand: 32 fp4 = 16 bytes per lane
using i32x8_t = opus::vector_t<opus::i32_t, 8>;
using sf8_t = opus::vector_t<opus::u32_t, 2>;      // the 8 scale bytes of one row and K tile

// ===========================================================================
// A / B: global -> LDS (LDS-DMA) -> VGPR
// ===========================================================================
// Create layout for loading A (or B) matrix from global memory: one LDS-DMA issue copies an 8-row x 128 B chunk (lane l
// lands at l * 16 B), so slot s of row r fetches the 16 B piece s ^ ((r >> 1) & 3) (conflict-free ds_read_b128 below).
template<class T>
__device__ inline auto make_layout_ga(int lane_id, int stride_a) {
    constexpr int threads_k = T::B_K_BYTES / T::VEC_A;

    constexpr auto ga_block_shape = opus::make_tuple(
        opus::number<T::smem_sub>{},
        opus::number<threads_k>{},
        opus::number<T::VEC_A>{});

    constexpr auto ga_block_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    const int lane_id_m = lane_id / threads_k;

    return opus::make_layout<T::VEC_A>(
        ga_block_shape,
        opus::unfold_x_stride(ga_block_dim, ga_block_shape, opus::tuple{stride_a, 1_I}),
        opus::unfold_p_coord(ga_block_dim, opus::tuple{lane_id_m, (lane_id % threads_k) ^ ((lane_id_m >> 1) & 3)}));
}
// Create layout for loading B matrix from global memory (the same 8-row chunks as A).
template<class T> __device__ inline auto make_layout_gb(int lane_id, int stride_b) { return make_layout_ga<T>(lane_id, stride_b); }

// Create layout for storing A matrix to shared memory: chunk (t, r) of a ring step = K tile t (KS > 1), rows [8 r, 8 r + 8).
template<class T>
__device__ inline constexpr auto make_layout_sa() {
    return opus::make_layout<-1>(opus::make_tuple(opus::number<T::KS>{}, opus::number<T::A_CHUNKS>{}),
                                 opus::make_tuple(opus::number<T::A_TILE_BYTES>{}, opus::number<T::CHUNK_BYTES>{}));
}
// Create layout for storing B matrix to shared memory (after the KS A tiles of the stage).
template<class T>
__device__ inline constexpr auto make_layout_sb() {
    return opus::make_layout<-1>(opus::make_tuple(opus::number<T::KS>{}, opus::number<T::B_CHUNKS>{}),
                                 opus::make_tuple(opus::number<T::B_TILE_BYTES>{}, opus::number<T::CHUNK_BYTES>{}));
}
// Create layout for the wave-uniform global offset of A chunk (t, r) (passed as the buffer soffset).
template<class T>
__device__ inline auto make_layout_gas(int stride_a) {
    return opus::make_layout<-1>(opus::make_tuple(opus::number<T::KS>{}, opus::number<T::A_CHUNKS>{}),
                                 opus::make_tuple(opus::number<T::B_K_BYTES>{}, T::smem_sub * stride_a));
}
// Create layout for the wave-uniform global offset of B chunk (t, r).
template<class T>
__device__ inline auto make_layout_gbs(int stride_b) {
    return opus::make_layout<-1>(opus::make_tuple(opus::number<T::KS>{}, opus::number<T::B_CHUNKS>{}),
                                 opus::make_tuple(opus::number<T::B_K_BYTES>{}, T::smem_sub * stride_b));
}

// Create layout for reading A (W = 1, E = E_M) or B (W = T_N, E = E_N) from shared memory to registers: lane holds row
// 16 (w, e) + lane % 16, K bytes [64 kk + 16 (lane / 16), +16) of the K tile of its wave group (wave_id_k); offsets[e * E_K + kk].
template<class T, int W, int E>
__device__ inline auto make_layout_r(int lane_id, int wave_id, int wave_id_k) {
    constexpr auto r_block_shape = opus::make_tuple(
        opus::number<T::T_K>{},
        opus::number<W>{},
        opus::number<E>{},
        opus::number<T::W_M / T::smem_sub>{},
        opus::number<T::smem_sub>{},
        opus::number<T::E_K>{},
        opus::number<T::WARP_SIZE / T::W_M>{},
        opus::number<T::VEC_A>{});

    constexpr auto r_block_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));

    const int lane_id_m = lane_id % T::W_M;
    const int r = lane_id_m % T::smem_sub;

    return opus::make_layout<T::VEC_A>(
        r_block_shape,
        opus::unfold_x_stride(r_block_dim, r_block_shape, opus::tuple{opus::number<T::CHUNK_BYTES>{}, 1_I}),
        opus::unfold_p_coord(r_block_dim, opus::tuple{wave_id_k, wave_id, lane_id_m / T::smem_sub, r, (lane_id / T::W_M) ^ ((r >> 1) & 3)}));
}
template<class T> __device__ inline auto make_layout_ra(int lane_id, int wave_id_k) { return make_layout_r<T, 1, T::E_M>(lane_id, 0, wave_id_k); }
template<class T> __device__ inline auto make_layout_rb(int lane_id, int wave_id_n, int wave_id_k) { return make_layout_r<T, T::T_N, T::E_N>(lane_id, wave_id_n, wave_id_k); }

// ===========================================================================
// Scales: row-major [rows][K/32], global -> LDS (block-wide 4-byte LDS-DMA) -> VGPR
// ===========================================================================
// Create layout for loading scale factors from global memory: one issue copies 32 rows x 8 B of a K tile, lane l copies row
// l / 2, bytes [4 (l % 2), +4) (lands at l * 4 B = the [row][8 B] image).  Rows / K tile go into the buffer soffset.
template<class T>
__device__ inline auto make_layout_gsf(int lane_id, int stride_sf) {
    constexpr int halves = T::SF_K_TILE_BYTES / T::VEC_SF_DMA;

    constexpr auto gsf_block_shape = opus::make_tuple(
        opus::number<T::SF_DMA_ROWS>{},
        opus::number<halves>{},
        opus::number<T::VEC_SF_DMA>{});

    constexpr auto gsf_block_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout<T::VEC_SF_DMA>(
        gsf_block_shape,
        opus::unfold_x_stride(gsf_block_dim, gsf_block_shape, opus::tuple{stride_sf, 1_I}),
        opus::unfold_p_coord(gsf_block_dim, opus::tuple{lane_id / halves, lane_id % halves}));
}
template<class T> __device__ inline auto make_layout_gsfa(int lane_id, int stride_sfa) { return make_layout_gsf<T>(lane_id, stride_sfa); }
template<class T> __device__ inline auto make_layout_gsfb(int lane_id, int stride_sfb) { return make_layout_gsf<T>(lane_id, stride_sfb); }

// Create layout for storing scale chunk (t, q) to shared memory: K tile t, 32-row chunk q (A rows for q < SFA_CHUNKS, then
// B rows; offset relative to the stage's A scale tiles, the B scale tiles follow them).
template<class T>
__device__ inline constexpr int make_offset_ssf(int t, int q) {
    return q < T::SFA_CHUNKS ? t * T::SFA_TILE_BYTES + q * T::SF_CHUNK_BYTES
                             : (T::STAGE_SFB_OFF - T::STAGE_SFA_OFF) + t * T::SFB_TILE_BYTES + (q - T::SFA_CHUNKS) * T::SF_CHUNK_BYTES;
}

// Create layout for reading scales from shared memory: row 16 (w, e) + lane % 16 of the K tile of wave group wave_id_k,
// 8 bytes; offsets[e].  A: W = 1, E = E_M, tile SFA_TILE_BYTES; B: W = T_N, E = E_N, tile SFB_TILE_BYTES.
template<class T, int W, int E, int TILE_BYTES>
__device__ inline auto make_layout_rsf(int lane_id, int wave_id, int wave_id_k) {
    constexpr auto rsf_block_shape = opus::make_tuple(
        opus::number<W>{},
        opus::number<E>{},
        opus::number<T::W_M>{},
        opus::number<T::VEC_SF>{});

    constexpr auto rsf_block_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}));

    auto u = opus::make_layout<T::VEC_SF>(
        rsf_block_shape,
        opus::unfold_x_stride(rsf_block_dim, rsf_block_shape, opus::tuple{opus::number<T::VEC_SF>{}, 1_I}),
        opus::unfold_p_coord(rsf_block_dim, opus::tuple{wave_id, lane_id % T::W_M}));
    opus::static_for<E>([&](auto e) { u.offsets[e.value] += wave_id_k * TILE_BYTES; });   // K tile of the wave group
    return u;
}
template<class T> __device__ inline auto make_layout_rsfa(int lane_id, int wave_id_k) { return make_layout_rsf<T, 1, T::E_M, T::SFA_TILE_BYTES>(lane_id, 0, wave_id_k); }
template<class T> __device__ inline auto make_layout_rsfb(int lane_id, int wave_id_n, int wave_id_k) { return make_layout_rsf<T, T::T_N, T::E_N, T::SFB_TILE_BYTES>(lane_id, wave_id_n, wave_id_k); }

// Register side: byte g = lane / 16 (kk 0) and 4 + g (kk 1) of the row's 8 bytes -> bytes 0 / 1 (v_perm), opsel = kk.
__device__ __forceinline__ unsigned sf_sel(int lane_id) { const unsigned g = lane_id / 16; return g | ((4u + g) << 8) | 0x0c0c0000u; }
__device__ __forceinline__ unsigned sf_pack(const sf8_t& v, unsigned sel) { return __builtin_amdgcn_perm(v[1], v[0], sel); }

// ===========================================================================
// C
// ===========================================================================
// Create layout for storing C to global memory: lane holds row 16 e + lane % 16, columns 16 n + 4 (lane / 16) .. +3 of
// its wave's columns; offsets[e * E_N + n] (elements).  Also the layout of the fp32 split-K partial tile.
template<class T>
__device__ inline auto make_layout_gc(int lane_id, int stride_c) {
    constexpr auto gc_block_shape = opus::make_tuple(
        opus::number<T::E_M>{},
        opus::number<T::W_M>{},
        opus::number<T::E_N>{},
        opus::number<T::WARP_SIZE / T::W_M>{},
        opus::number<T::VEC_C>{});

    constexpr auto gc_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout<T::VEC_C>(
        gc_block_shape,
        opus::unfold_x_stride(gc_block_dim, gc_block_shape, opus::tuple{stride_c, 1_I}),
        opus::unfold_p_coord(gc_block_dim, opus::tuple{lane_id % T::W_M, lane_id / T::W_M}));
}

// One __shared__ object per ring stage (kept explicit instead of one opus smem arena): the compiler's waitcnt pass then
// knows an LDS-DMA into stage j cannot alias a ds_read of stage i, instead of waiting for every in-flight copy before
// each read (one array = everything aliases).  LDS accesses below use address_space(3) pointers derived from these
// objects (not opus::smem, whose integer round trip loses the object), for the same reason.
template<int I, int BYTES>
__device__ __forceinline__ char* lds_stage() {
    __shared__ __attribute__((aligned(16))) char buf[BYTES];
    return buf;
}
__device__ __forceinline__ OPUS_LDS_ADDR char* lds_ptr(char* p) { return (OPUS_LDS_ADDR char*)p; }

// sched_group_barrier masks (LLVM AMDGPU)
constexpr int SCHED_MFMA = 0x008;
constexpr int SCHED_VMEM_READ = 0x820;   // 0x020 VMEM read | 0x800 LDS DMA (newer LLVM classifies buffer_load ... lds as LDS DMA)
constexpr int SCHED_DS_READ = 0x100;

template<int MASK, int CNT = 1>
__device__ inline void sched_group() {
    if constexpr (CNT > 0) __builtin_amdgcn_sched_group_barrier(MASK, CNT, 0);
}

// One scaled fp4 MFMA with B as srcA: C fragment row = lane % 16, columns 4 (lane / 16) .. +3.
template<int OPSEL_B, int OPSEL_A, class CF>
__device__ __forceinline__ void mma_scale_one(CF& c, const fp4x32_t& b, const fp4x32_t& a, unsigned v_sfb, unsigned v_sfa) {
    using mma = opus::mfma_f32_16x16x128_fp4_fp4;   // fp4 x fp4 -> fp32, 16x16x128 (only the low 4 dwords of srcA/B are read)
    const i32x8_t b8 = __builtin_shufflevector(b, b, 0, 1, 2, 3, -1, -1, -1, -1);
    const i32x8_t a8 = __builtin_shufflevector(a, a, 0, 1, 2, 3, -1, -1, -1, -1);
    c = mma{}(b8, a8, c, static_cast<int>(v_sfb), static_cast<int>(v_sfa), opus::number<OPSEL_B>{}, opus::number<OPSEL_A>{});
}

// ===========================================================================
// Kernel body
// ===========================================================================
template<class Traits>
__device__ __forceinline__ void gemm_a4w4_mxfp4_nt_small_noshuffle_body(const opus_gemm_mxfp4_small_noshuffle_kargs_gfx950& kargs) {
    using namespace opus;
    using T = opus::remove_cvref_t<Traits>;
    using D_AB = opus::u8_t;     // fp4x2 bytes
    using D_SF = opus::u8_t;     // e8m0 bytes
    using D_ACC = opus::fp32_t;
    using D_C = opus::bf16_t;
    using CFrag = opus::vector_t<D_ACC, T::VEC_C>;
    constexpr int D = T::STAGES, G = T::VMEM_PER_STEP;

    const int lane_id = static_cast<int>(opus::lane_id());
    const int wave_id = __builtin_amdgcn_readfirstlane(thread_id_x() / T::WARP_SIZE);
    const int wave_id_n = wave_id % T::T_N, wave_id_k = wave_id / T::T_N;
    const int split_k = kargs.split_k;
    const int row0 = block_id_y() * T::B_M;
    const int col0 = block_id_x() * T::B_N;                           // the block's first column (B / scale copies)
    const int col0_w = col0 + wave_id_n * T::WAVE_N;                   // this wave's first column (C)
    const int batch_id = block_id_z() / split_k, k_split = block_id_z() % split_k;
    // S > 1: read the workspace slot now, so the (host-coherent) load is long done when the partial tile is written.
    float* ws = nullptr;
    if (split_k > 1) ws = static_cast<float*>(kargs.ws_handle->ptr);
    const int k_tiles = kargs.k / T::B_K, k_per = (k_tiles + split_k - 1) / split_k;
    const int k_begin = k_split * k_per;
    const int k_end = k_tiles < k_begin + k_per ? k_tiles : k_begin + k_per;
    const int stride_b = kargs.k / 2;   // bytes per B row

    // M / N tails: rows / columns past the matrix lie outside the buffer ranges (copies write 0).
    auto clampb = [](long long v) { return v <= 0 ? 0u : (v > 0x7fffffffLL ? 0x7fffffffu : static_cast<unsigned>(v)); };
    const D_AB* pa = reinterpret_cast<const D_AB*>(kargs.ptr_a) + batch_id * kargs.stride_a_batch + static_cast<long long>(row0) * kargs.stride_a;
    const D_AB* pb = reinterpret_cast<const D_AB*>(kargs.ptr_b) + batch_id * kargs.stride_b_batch + static_cast<long long>(col0) * stride_b;
    const D_SF* psfa = reinterpret_cast<const D_SF*>(kargs.ptr_sfa) + batch_id * kargs.stride_sfa_batch + static_cast<long long>(row0) * kargs.stride_sfa;
    const D_SF* psfb = reinterpret_cast<const D_SF*>(kargs.ptr_sfb) + batch_id * kargs.stride_sfb_batch + static_cast<long long>(col0) * kargs.stride_sfb;
    const unsigned a_bytes = clampb(static_cast<long long>(kargs.m - row0) * kargs.stride_a);
    const unsigned b_bytes = clampb(static_cast<long long>(kargs.n - col0) * stride_b);
    const unsigned sfa_bytes = clampb(static_cast<long long>(kargs.m - row0) * kargs.stride_sfa);
    const unsigned sfb_bytes = clampb(static_cast<long long>(kargs.n - col0) * kargs.stride_sfb);
    const unsigned sel = sf_sel(lane_id);

    const auto u_ga = make_layout_ga<T>(lane_id, kargs.stride_a);
    const auto u_gb = make_layout_gb<T>(lane_id, stride_b);
    constexpr auto u_sa = make_layout_sa<T>();
    constexpr auto u_sb = make_layout_sb<T>();
    const auto u_gas = make_layout_gas<T>(kargs.stride_a);
    const auto u_gbs = make_layout_gbs<T>(stride_b);
    const auto u_gsfa = make_layout_gsfa<T>(lane_id, kargs.stride_sfa);
    const auto u_gsfb = make_layout_gsfb<T>(lane_id, kargs.stride_sfb);
    const auto u_ra = make_layout_ra<T>(lane_id, wave_id_k);
    const auto u_rb = make_layout_rb<T>(lane_id, wave_id_n, wave_id_k);
    const auto u_rsfa = make_layout_rsfa<T>(lane_id, wave_id_k);
    const auto u_rsfb = make_layout_rsfb<T>(lane_id, wave_id_n, wave_id_k);

    // This wave's copies of a ring step, spread round-robin over the 4 waves (chunk index clamped: with fewer chunks than
    // waves the last waves repeat the last chunk).  Everything but the K tile is fixed per wave, so it is computed once here
    // (wave-uniform: SGPRs; the LDS destination goes into M0): K tile t of the step, LDS offset in the stage, global soffset.
    auto chunk = [&](int i, int insts, int chunks) {
        const int c0 = wave_id * insts + i;
        return __builtin_amdgcn_readfirstlane(c0 < chunks ? c0 : chunks - 1);
    };
    int a_t[T::a_buffer_load_insts], a_lds[T::a_buffer_load_insts], a_gso[T::a_buffer_load_insts];
    static_for<T::a_buffer_load_insts>([&](auto i) {
        const int c = chunk(i.value, T::a_buffer_load_insts, T::A_DMA_CHUNKS);
        const int t = T::KS == 1 ? 0 : c / T::A_CHUNKS, r = T::KS == 1 ? c : c % T::A_CHUNKS;   // K tile t, rows [8 r, 8 r + 8)
        a_t[i.value] = t; a_lds[i.value] = u_sa(t, r); a_gso[i.value] = u_gas(t, r);
    });
    int b_t[T::b_buffer_load_insts], b_lds[T::b_buffer_load_insts], b_gso[T::b_buffer_load_insts];
    static_for<T::b_buffer_load_insts>([&](auto i) {
        const int c = chunk(i.value, T::b_buffer_load_insts, T::B_DMA_CHUNKS);
        const int t = T::KS == 1 ? 0 : c / T::B_CHUNKS, r = T::KS == 1 ? c : c % T::B_CHUNKS;
        b_t[i.value] = t; b_lds[i.value] = T::STAGE_B_OFF + u_sb(t, r); b_gso[i.value] = u_gbs(t, r);
    });
    // Scale chunk q of K tile t: A rows for q < SFA_CHUNKS, then B rows (source, range and lane offset chosen per wave).
    const D_SF* sf_p[T::sf_buffer_load_insts];
    unsigned sf_bytes[T::sf_buffer_load_insts];
    int sf_t[T::sf_buffer_load_insts], sf_lds[T::sf_buffer_load_insts], sf_gso[T::sf_buffer_load_insts], sf_vo[T::sf_buffer_load_insts];
    static_for<T::sf_buffer_load_insts>([&](auto i) {
        const int c = chunk(i.value, T::sf_buffer_load_insts, T::SF_DMA_CHUNKS);
        const int t = T::KS == 1 ? 0 : c / T::SF_CHUNKS, q = T::KS == 1 ? c : c % T::SF_CHUNKS;
        const bool isa = q < T::SFA_CHUNKS;
        sf_p[i.value] = isa ? psfa : psfb;
        sf_bytes[i.value] = isa ? sfa_bytes : sfb_bytes;
        sf_t[i.value] = t;
        sf_lds[i.value] = T::STAGE_SFA_OFF + make_offset_ssf<T>(t, q);
        sf_gso[i.value] = (isa ? q : q - T::SFA_CHUNKS) * T::SF_DMA_ROWS * (isa ? kargs.stride_sfa : kargs.stride_sfb) + t * T::SF_K_TILE_BYTES;
        sf_vo[i.value] = isa ? u_gsfa.offsets[0] : u_gsfb.offsets[0];
    });

    // All copies of ring step `slot` (K tiles kt .. kt + KS - 1).  K tiles past k_end read through zero-range buffers, so the
    // copy count per step is fixed (counted vmcnt) and the loop has no branch.
    auto issue = [&](auto slot_i, int kt) {
        constexpr int slot = decltype(slot_i)::value;
        char* const stage = lds_stage<slot, T::STAGE_BYTES>();
        static_for<T::a_buffer_load_insts>([&](auto i) {
            auto g_a = make_gmem(pa, kt + a_t[i.value] < k_end ? a_bytes : 0u);
            g_a.template _async_load<T::VEC_A>(lds_ptr(stage + a_lds[i.value]), u_ga.offsets[0], a_gso[i.value] + kt * T::B_K_BYTES);
        });
        static_for<T::b_buffer_load_insts>([&](auto i) {
            auto g_b = make_gmem(pb, kt + b_t[i.value] < k_end ? b_bytes : 0u);
            g_b.template _async_load<T::VEC_B>(lds_ptr(stage + b_lds[i.value]), u_gb.offsets[0], b_gso[i.value] + kt * T::B_K_BYTES);
        });
        static_for<T::sf_buffer_load_insts>([&](auto i) {
            auto g_sf = make_gmem(sf_p[i.value], kt + sf_t[i.value] < k_end ? sf_bytes[i.value] : 0u);
            g_sf.template _async_load<T::VEC_SF_DMA>(lds_ptr(stage + sf_lds[i.value]), sf_vo[i.value], sf_gso[i.value] + kt * T::SF_K_TILE_BYTES);
        });
    };
    // This wave's operands and packed scales of ring step `slot`.
    struct operands { fp4x32_t a[T::E_K][T::E_M]; fp4x32_t b[T::E_K][T::E_N]; sf8_t sfa[T::E_M]; sf8_t sfb[T::E_N]; };
    auto read = [&](operands& v, auto slot_i) {
        OPUS_LDS_ADDR char* const stage = lds_ptr(lds_stage<decltype(slot_i)::value, T::STAGE_BYTES>());
        static_for<T::E_K>([&](auto kk) {
            static_for<T::E_M>([&](auto e) {
                v.a[kk.value][e.value] = *reinterpret_cast<OPUS_LDS_ADDR const fp4x32_t*>(stage + u_ra.offsets[e.value * T::E_K + kk.value]);
            });
            static_for<T::E_N>([&](auto n) {
                v.b[kk.value][n.value] = *reinterpret_cast<OPUS_LDS_ADDR const fp4x32_t*>(stage + T::STAGE_B_OFF + u_rb.offsets[n.value * T::E_K + kk.value]);
            });
        });
        static_for<T::E_M>([&](auto e) { v.sfa[e.value] = *reinterpret_cast<OPUS_LDS_ADDR const sf8_t*>(stage + T::STAGE_SFA_OFF + u_rsfa.offsets[e.value]); });
        static_for<T::E_N>([&](auto n) { v.sfb[n.value] = *reinterpret_cast<OPUS_LDS_ADDR const sf8_t*>(stage + T::STAGE_SFB_OFF + u_rsfb.offsets[n.value]); });
    };

    CFrag v_c[T::E_M][T::E_N];
    static_for<T::E_M>([&](auto e) { static_for<T::E_N>([&](auto n) { clear(v_c[e.value][n.value]); }); });
    const int steps = k_end > k_begin ? (k_end - k_begin + T::KS - 1) / T::KS : 0;   // ring steps (KS K tiles each)

    // Ring step t lives in slot t % STAGES.  Prologue: issue steps 0 .. STAGES-1, wait for step 0, read step 0.
    // The waits are asm with a "memory" clobber (vmcnt and lgkmcnt in one instruction, and a compiler barrier for the
    // LDS reads that follow), counted in VMEM_PER_STEP units: opus::s_waitcnt_vmcnt is not a compiler memory barrier.
    static_for<D>([&](auto t) { issue(t, k_begin + T::KS * decltype(t)::value); __builtin_amdgcn_sched_barrier(0); });
    operands v_cur;   // operands of the current step
    // The workspace-slot s_load (S > 1) must not be pending in the loop: the waitcnt pass cannot see the asm waits and,
    // with an SMEM load outstanding, would turn its ds_read waits into lgkmcnt(0).  Drained here (builtin: visible to it).
    s_waitcnt_lgkmcnt(0_I);
    asm volatile("s_waitcnt vmcnt(%0)" :: "n"((D - 1) * G) : "memory");
    sync_threads();
    read(v_cur, 0_I);
    __builtin_amdgcn_sched_barrier(0);
#pragma clang loop unroll(disable)
    for (int i0 = 0; i0 < steps; i0 += D) {
        static_for<D>([&](auto u) {
            constexpr int s = decltype(u)::value, sn = (s + 1) % D;
            const int kt = k_begin + T::KS * (i0 + s);
            // Step t+1 landed (D-2 younger steps in flight) and this wave's reads of step t are done; the barrier makes step
            // t+1 visible and frees step t's stage (every wave has read it) for the copies of step t+D.
            asm volatile("s_waitcnt vmcnt(%0) lgkmcnt(0)" :: "n"((D - 2) * G) : "memory");
            sync_threads();
            __builtin_amdgcn_sched_barrier(0);
            operands v_next;
            read(v_next, number<sn>{});
            unsigned sfa[T::E_M], sfb[T::E_N];
            static_for<T::E_M>([&](auto e) { sfa[e.value] = sf_pack(v_cur.sfa[e.value], sel); });
            static_for<T::E_N>([&](auto n) { sfb[n.value] = sf_pack(v_cur.sfb[n.value], sel); });
            static_for<T::E_K>([&](auto kk) { static_for<T::E_M>([&](auto e) { static_for<T::E_N>([&](auto n) {
                constexpr int K = decltype(kk)::value, E = decltype(e)::value, N = decltype(n)::value;
                mma_scale_one<K, K>(v_c[E][N], v_cur.b[K][N], v_cur.a[K][E], sfb[N], sfa[E]);
            }); }); });
            issue(number<s>{}, kt + T::KS * D);
            // Schedule (kept explicit): the next step's ds_reads first, then MFMA / VMEM alternating.
            constexpr int MFMAS = T::E_K * T::E_M * T::E_N;
            sched_group<SCHED_DS_READ, T::DS_READS_PER_STEP>();
            static_for<MFMAS>([&](auto m) {
                sched_group<SCHED_MFMA>();
                sched_group<SCHED_VMEM_READ, (decltype(m)::value < G)>();
            });
            sched_group<SCHED_VMEM_READ, (G > MFMAS ? G - MFMAS : 0)>();
            __builtin_amdgcn_sched_barrier(0);
            v_cur = v_next;
        });
    }
    s_waitcnt_vmcnt(0_I);

    if constexpr (T::KS > 1) {   // wave groups 1..KS-1 park their partial tiles in LDS, group 0 adds them
        constexpr int FRAGS = T::E_M * T::E_N;
        __shared__ __attribute__((aligned(16))) char red[T::RED_BYTES];
        auto s_red = make_smem(reinterpret_cast<D_ACC*>(red));
        auto red_off = [&](int g, int f) { return (((g * T::T_N + wave_id_n) * FRAGS + f) * T::WARP_SIZE + lane_id) * T::VEC_C; };
        if (wave_id_k > 0)
            static_for<FRAGS>([&](auto f) { store<T::VEC_C>(s_red, v_c[f.value / T::E_N][f.value % T::E_N], red_off(wave_id_k - 1, f.value)); });
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
        sync_threads();
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
        if (wave_id_k > 0) return;
        static_for<T::KS - 1>([&](auto g) { static_for<FRAGS>([&](auto f) {
            v_c[f.value / T::E_N][f.value % T::E_N] += load<T::VEC_C>(s_red, red_off(g.value, f.value));
        }); });
    }

    if (split_k > 1) {   // fp32 partial tile -> workspace [S][batch][padded_M][padded_N]; splitk_reduce_kernel writes C
        const int padded_m = (kargs.m + T::B_M - 1) / T::B_M * T::B_M, padded_n = (kargs.n + T::B_N - 1) / T::B_N * T::B_N;
        auto g_w = make_gmem(ws + (static_cast<long long>(k_split * kargs.batch + batch_id) * padded_m + row0) * padded_n + col0_w,
                             static_cast<unsigned>((T::B_M * padded_n) * sizeof(float)));
        const auto u_gw = make_layout_gc<T>(lane_id, padded_n);
        static_for<T::E_M * T::E_N>([&](auto f) { store<T::VEC_C>(g_w, v_c[f.value / T::E_N][f.value % T::E_N], u_gw.offsets[f.value]); });
        return;
    }
    auto g_c = make_gmem(reinterpret_cast<D_C*>(kargs.ptr_c) + batch_id * kargs.stride_c_batch + static_cast<long long>(row0) * kargs.stride_c + col0_w,
                         clampb(static_cast<long long>(kargs.m - row0) * kargs.stride_c * 2));
    const auto u_gc = make_layout_gc<T>(lane_id, kargs.stride_c);
    static_for<T::E_M>([&](auto e) { static_for<T::E_N>([&](auto n) {
        if (col0_w + 16 * n.value + 4 * (lane_id / 16) < kargs.n)   // N tail: whole 4-column groups (N % 8 == 0)
            store<T::VEC_C>(g_c, opus::cast<D_C>(v_c[e.value][n.value]), u_gc.offsets[e.value * T::E_N + n.value]);
    }); });
}

}   // namespace opus_mxfp4_small_noshuffle

// grid = ((N + B_N - 1) / B_N, (M + B_M - 1) / B_M, batch * split_k), block = 256.
template<class Traits>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, 1) void gemm_a4w4_mxfp4_nt_small_noshuffle_kernel(opus_gemm_mxfp4_small_noshuffle_kargs_gfx950 kargs) {
#if defined(__gfx950__)
    opus_mxfp4_small_noshuffle::gemm_a4w4_mxfp4_nt_small_noshuffle_body<Traits>(kargs);
#endif   // __gfx950__
}
