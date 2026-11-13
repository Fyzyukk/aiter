#pragma once

#include <opus/hip_minimal.hpp>
#include <opus/opus.hpp>

#include "opus_gemm_traits_a4w4_mxfp4_nt_gfx950.cuh"

#ifndef AB_AUX   // cache policy of the A / B LDS-DMA loads
#define AB_AUX 0
#endif

using opus::operator""_I;

// ===========================================================================
// A: global -> LDS -> VGPR
// ===========================================================================
template<class T>
__device__ inline auto make_layout_ga(int lane_id, int wave_id_m, int wave_id_n, int stride_a) {
    // Wave (P0, P1) copies chunks (P0 * 2 + P1) * 4 + i (i = ih * 2 + il): its 4 issues land in
    // adjacent LDS chunks (1056 B apart, an immediate offset), with the same rows per chunk as
    // before: chunk c holds rows (c / 2) * 16 + 2 * (lane / 8) + c % 2.
    constexpr int threads_k = T::B_K_BYTES / T::VEC_A;                 // 8
    static_assert(threads_k == T::B_K / T::VEC_A_ELEMS && T::T_M == 2 && T::T_N == 2);
    static_assert(T::HALF_B_M / (T::BLOCK_SIZE / threads_k) == T::a_buffer_load_insts);
    constexpr auto shape = opus::make_tuple(
        opus::number<2>{},                                             // p P0
        opus::number<2>{},                                             // p P1
        opus::number<T::a_buffer_load_insts / 2>{},                   // y ih
        opus::number<T::WARP_SIZE / threads_k>{},                      // p lane / 8
        opus::number<2>{},                                             // y il
        opus::number<threads_k>{},                                     // p lane % 8
        opus::number<T::VEC_A>{});                                     // y 16 bytes
    constexpr auto dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}, opus::p_dim{}, opus::y_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));
    return opus::make_layout<T::VEC_A>(
        shape,
        opus::unfold_x_stride(dim, shape, opus::tuple{stride_a, 1_I}),
        opus::unfold_p_coord(dim, opus::tuple{wave_id_n, wave_id_m, lane_id / threads_k, (lane_id % threads_k) ^ ((((lane_id / threads_k) >> 1) & 1) << 1)}));
}

template<class T>
__device__ inline auto make_layout_sa(int wave_id_m, int wave_id_n) {
    // Chunk (P0 * 2 + P1) * 4 + issue (see make_layout_ga).
    constexpr auto shape = opus::make_tuple(
        opus::number<2>{},                                             // p P0
        opus::number<2>{},                                             // p P1
        opus::number<T::a_buffer_load_insts>{},                        // y issue
        opus::number<T::VEC_A>{});                                     // y 16 bytes
    constexpr auto dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}),
        opus::make_tuple(opus::y_dim{}));
    return opus::make_layout(
        shape,
        opus::unfold_x_stride(dim, shape, opus::tuple{opus::number<T::smem_linear_wave_a + T::smem_padding>{}, 1_I}),
        opus::unfold_p_coord(dim, opus::tuple{wave_id_n, wave_id_m}));
}

template<class T>
__device__ inline auto make_layout_ra(int lane_id, int wave_id_m) {
    static_assert(T::smem_sub_a * T::T_M == T::W_M && T::T_M == 2);
    constexpr int reads_per_operand = T::W_M * T::W_K_BYTES / T::WARP_SIZE / T::VEC_A;   // 1 b128 per fp4 operand
    static_assert(reads_per_operand == 1);
    static_assert(T::WARP_SIZE / T::W_M * T::VEC_A_ELEMS == T::W_K);   // 4 lane groups x 32 fp4 = K 128

    constexpr auto ra_block_shape = opus::make_tuple(
        opus::number<T::E_M>{},                                        // 4  y M repeat
        opus::number<T::T_M>{},                                        // 2  p wave_m
        opus::number<T::T_M>{},                                        // 2  p r % 2
        opus::number<T::W_M / T::T_M>{},                               // 8  p r / 2
        opus::number<T::E_K>{},                                        // 2  p kk, fixed to 0 (keeps the 128 B row stride); kk -> + kk * W_K_BYTES
        opus::number<reads_per_operand>{},                             // 1  y b128 per operand
        opus::number<T::WARP_SIZE / T::W_M>{},                         // 4  p g = lane / 16
        opus::number<T::VEC_A>{});                                     // 16 y bytes
    constexpr auto ra_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));
    const int lane_m = lane_id % T::W_M;
    return opus::make_layout<T::VEC_A>(
        ra_block_shape,
        opus::unfold_x_stride(ra_block_dim, ra_block_shape, opus::tuple{opus::number<T::smem_linear_wave_a + T::smem_padding>{}, 1_I}),
        opus::unfold_p_coord(ra_block_dim, opus::tuple{wave_id_m, lane_m % T::T_M, lane_m / T::T_M, 0_I, (lane_id / T::W_M) ^ (((lane_m >> 2) & 1) << 1)}));
}

template<class T>
__device__ inline auto make_layout_gb(int lane_id, int wave_id_m, int wave_id_n, int stride_b) {
    // bs: B in aiter shuffle_weight(w, layout=(16, 16)) layout: byte (n, kb) at (n/16)*16*Kb + (kb/32)*512 +
    // ((kb/16)%2)*256 + (n%16)*16 + kb%16, Kb = stride_b.  One 16-row block x one 128 B K tile = 2048 contiguous
    // bytes; its K halves (kk = 0, 1) are 1 KB each.  Wave (P0, P1) issue i = ih * 2 + il copies 16-row block
    // (P0 * 2 + P1) * 2 + ih, K half il, as one contiguous 1 KB, into LDS chunk (P0 * 2 + P1) * 4 + i (as PR 6696):
    // chunk = 2 * block + kk holds [g 4][n 16][16 B].
    static_assert(T::T_M == 2 && T::T_N == 2 && T::W_N == 16 && T::B_K_BYTES == 128);
    constexpr auto shape = opus::make_tuple(
        opus::number<2>{},                                             // p P0 (wave_m)
        opus::number<2>{},                                             // p P1 (wave_n)
        opus::number<T::b_buffer_load_insts / 2>{},                   // y ih: 16-row block
        opus::number<2>{},                                             // y il: K half (1 KB)
        opus::number<T::WARP_SIZE>{},                                  // p lane
        opus::number<T::VEC_B>{});                                     // y 16 bytes
    constexpr auto dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}),
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));
    return opus::make_layout<T::VEC_B>(
        shape,
        opus::unfold_x_stride(dim, shape, opus::tuple{T::W_N * stride_b, 1_I}),
        opus::unfold_p_coord(dim, opus::tuple{wave_id_m, wave_id_n, lane_id}));
}

template<class T>
__device__ inline auto make_layout_sb(int wave_id_m, int wave_id_n) {
    // Chunk (P0 * 2 + P1) * 4 + issue (see make_layout_gb).
    constexpr auto shape = opus::make_tuple(
        opus::number<2>{},                                             // p P0
        opus::number<2>{},                                             // p P1
        opus::number<T::b_buffer_load_insts>{},                        // y issue
        opus::number<T::VEC_A>{});                                     // y 16 bytes
    constexpr auto dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}),
        opus::make_tuple(opus::y_dim{}));
    return opus::make_layout(
        shape,
        opus::unfold_x_stride(dim, shape, opus::tuple{opus::number<T::smem_linear_wave_a + T::smem_padding>{}, 1_I}),
        opus::unfold_p_coord(dim, opus::tuple{wave_id_m, wave_id_n}));
}

template<class T>
__device__ inline auto make_layout_rb(int lane_id, int wave_id_n) {
    // bs: repeat e of wave_n reads 16-row block 2e + wave_n, chunk (2e + wave_n) * 2 + kk (kk -> + kk * chunk stride
    // in load_b): lane reads chunk + 16 * lane = row lane % 16, K bytes [64 kk + 16 g, +16), g = lane / 16.  Each read
    // is one contiguous 1 KB: bank-conflict free.
    static_assert(T::W_N * T::W_K_BYTES / T::WARP_SIZE / T::VEC_B == 1 && T::T_N == 2);
    constexpr int P = T::smem_linear_wave_b + T::smem_padding;
    constexpr auto rb_block_shape = opus::make_tuple(
        opus::number<T::E_N>{},                                        // 4  y N repeat
        opus::number<T::T_N>{},                                        // 2  p wave_n
        opus::number<T::WARP_SIZE>{},                                  // 64 p lane
        opus::number<T::VEC_B>{});                                     // 16 y bytes
    constexpr auto rb_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));
    return opus::make_layout<T::VEC_B>(
        rb_block_shape,
        opus::unfold_x_stride(rb_block_dim, rb_block_shape, opus::tuple{opus::number<2 * P>{}, 1_I}),
        opus::unfold_p_coord(rb_block_dim, opus::tuple{wave_id_n, lane_id}));
}

template<class T>
__device__ inline auto make_layout_gsfa(int lane_id, int wave_id_m, int wave_id_n, int stride_sfa) {
    // Writer wave (wave_id_n, wave_id_m), lane (e = lane % 4, r = lane / 4) loads row wave_id_n * 128 + e * 32 +
    // wave_id_m * 16 + r: after the 4x4 byte transpose across e, lane (g, r) holds byte e = repeat e,
    // K group g -- the MFMA scale layout of wave (h = wave_id_n, w = wave_id_m).
    constexpr auto shape = opus::make_tuple(
        opus::number<2>{},                                             // p wave_id_n: half (128 rows)
        opus::number<4>{},                                             // p e (32 rows)
        opus::number<2>{},                                             // p wave_id_m (16 rows)
        opus::number<16>{},                                            // p r
        opus::number<T::VEC_GLOBAL_SCALE>{});                          // y 16 bytes (panel)
    constexpr auto dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}));
    static_assert(2 * 4 * 2 * 16 == T::B_M && T::B_M == T::B_N);
    return opus::make_layout<T::VEC_GLOBAL_SCALE>(
        shape,
        opus::unfold_x_stride(dim, shape, opus::tuple{stride_sfa, 1_I}),
        opus::unfold_p_coord(dim, opus::tuple{wave_id_n, lane_id % 4, wave_id_m, lane_id / 4}));   // lane = 4 r + e
}

template<class T>
__device__ inline auto make_layout_ssfa(int wave_id_m, int wave_id_n) {
    constexpr int wave_bytes = T::WARP_SIZE * T::VEC_GLOBAL_SCALE;    // 1024
    constexpr auto ssfa_block_shape = opus::make_tuple(
        opus::number<T::B_M / T::BLOCK_SIZE>{},                        // 1  y M repeat
        opus::number<T::T_N>{},                                        // 2  p wave_n
        opus::number<T::T_M>{},                                        // 2  p wave_m
        opus::number<T::VEC_GLOBAL_SCALE>{});                          // 16 y bytes
    constexpr auto ssfa_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}));
    return opus::make_layout<T::VEC_GLOBAL_SCALE>(
        ssfa_block_shape,
        opus::unfold_x_stride(ssfa_block_dim, ssfa_block_shape, opus::tuple{opus::number<wave_bytes>{}, 1_I}),
        opus::unfold_p_coord(ssfa_block_dim, opus::tuple{wave_id_n, wave_id_m}));
}

template<class T>
__device__ inline auto make_layout_gsfb(int lane_id, int wave_id_m, int wave_id_n, int stride_sfb) {
    // Writer wave (wave_id_m, wave_id_n), lane (e = lane % 4, r = lane / 4) loads row wave_id_m * 128 + e * 32 +
    // wave_id_n * 16 + r: after the 4x4 byte transpose across e, lane (g, r) holds byte e = repeat e,
    // K group g -- the MFMA scale layout of wave (h = wave_id_m, w = wave_id_n).
    constexpr auto shape = opus::make_tuple(
        opus::number<2>{},                                             // p wave_id_m: half (128 rows)
        opus::number<4>{},                                             // p e (32 rows)
        opus::number<2>{},                                             // p wave_id_n (16 rows)
        opus::number<16>{},                                            // p r
        opus::number<T::VEC_GLOBAL_SCALE>{});                          // y 16 bytes (panel)
    constexpr auto dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}));
    static_assert(2 * 4 * 2 * 16 == T::B_M && T::B_M == T::B_N);
    return opus::make_layout<T::VEC_GLOBAL_SCALE>(
        shape,
        opus::unfold_x_stride(dim, shape, opus::tuple{stride_sfb, 1_I}),
        opus::unfold_p_coord(dim, opus::tuple{wave_id_m, lane_id % 4, wave_id_n, lane_id / 4}));   // lane = 4 r + e
}

template<class T>
__device__ inline auto make_layout_ssfb(int wave_id_m, int wave_id_n) {
    constexpr int wave_bytes = T::WARP_SIZE * T::VEC_GLOBAL_SCALE;    // 1024
    constexpr auto ssfb_block_shape = opus::make_tuple(
        opus::number<T::B_N / T::BLOCK_SIZE>{},                        // 1  y N repeat
        opus::number<T::T_M>{},                                        // 2  p wave_m
        opus::number<T::T_N>{},                                        // 2  p wave_n
        opus::number<T::VEC_GLOBAL_SCALE>{});                          // 16 y bytes
    constexpr auto ssfb_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}));
    return opus::make_layout<T::VEC_GLOBAL_SCALE>(
        ssfb_block_shape,
        opus::unfold_x_stride(ssfb_block_dim, ssfb_block_shape, opus::tuple{opus::number<wave_bytes>{}, 1_I}),
        opus::unfold_p_coord(ssfb_block_dim, opus::tuple{wave_id_m, wave_id_n}));
}

template<class T>
__device__ inline auto make_layout_rsfa(int lane_id, int wave_id_m) {
    constexpr auto rsfa_block_shape = opus::make_tuple(
        opus::number<T::B_M / T::HALF_B_M>{},                          // 2  y half h   (128 rows)
        opus::number<T::WARP_SIZE / T::W_M>{},                         // 4  p g as e   (32 rows)
        opus::number<T::T_M>{},                                        // 2  p wave_m   (16 rows)
        opus::number<T::W_M>{},                                        // 16 p r        (1 row)
        opus::number<T::VEC_LDS_SCALE>{});                             // 8  y bytes
    constexpr auto rsfa_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}));
    return opus::make_layout<T::VEC_LDS_SCALE>(
        rsfa_block_shape,
        opus::unfold_x_stride(rsfa_block_dim, rsfa_block_shape, opus::tuple{opus::number<T::VEC_GLOBAL_SCALE>{}, 1_I}),
        opus::unfold_p_coord(rsfa_block_dim, opus::tuple{lane_id / T::W_M, wave_id_m, lane_id % T::W_M}));
}

template<class T>
__device__ inline auto make_layout_rsfb(int lane_id, int wave_id_n) {
    constexpr auto rsfb_block_shape = opus::make_tuple(
        opus::number<T::B_N / T::HALF_B_N>{},                          // 2  y half h   (128 rows)
        opus::number<T::WARP_SIZE / T::W_N>{},                         // 4  p g as e   (32 rows)
        opus::number<T::T_N>{},                                        // 2  p wave_n   (16 rows)
        opus::number<T::W_N>{},                                        // 16 p r        (1 row)
        opus::number<T::VEC_LDS_SCALE>{});                             // 8  y bytes
    constexpr auto rsfb_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}));
    return opus::make_layout<T::VEC_LDS_SCALE>(
        rsfb_block_shape,
        opus::unfold_x_stride(rsfb_block_dim, rsfb_block_shape, opus::tuple{opus::number<T::VEC_GLOBAL_SCALE>{}, 1_I}),
        opus::unfold_p_coord(rsfb_block_dim, opus::tuple{lane_id / T::W_N, wave_id_n, lane_id % T::W_N}));
}

__device__ inline unsigned int transpose_scale_dword_4x4(unsigned int packed, int scale_call) {
    const auto rows01_or_23 = __builtin_amdgcn_permlane16_swap(packed, packed, false, true);
    const unsigned int select_q01 = 0x06020400u + (scale_call & 1) * 0x01010101u;
    const unsigned int q01_or_q23 = __builtin_amdgcn_perm(rows01_or_23[1], rows01_or_23[0], select_q01);
    const auto rows0123 = __builtin_amdgcn_permlane32_swap(q01_or_q23, q01_or_q23, false, true);
    const unsigned int select_q = 0x05040100u + (scale_call >> 1) * 0x02020202u;
    return __builtin_amdgcn_perm(rows0123[1], rows0123[0], select_q);
}


// Like opus async_load(layout), but the per-issue LDS offsets (compile-time, from the static
// shape / stride of the LDS layout) go into the MUBUF immediate offset, so all issues share one
// LDS base (m0).  The immediate shifts the global address too, so it is subtracted there.
template<int VEC, class G, class LG, class LS>
__device__ inline void async_load_imm(G& g, void* smem_base, const LG& u_g, const LS& u_s, int s_os) {
    const auto gofs = opus::layout_to_offsets<VEC>(u_g);
    const auto sofs = opus::layout_to_offsets<VEC>(u_s);
    constexpr auto imm = opus::layout_imm_offsets_v<opus::remove_cvref_t<LS>, VEC>;
    constexpr int n = static_cast<int>(sizeof(imm) / sizeof(imm[0]));
    char* const base = reinterpret_cast<char*>(smem_base) + sofs[0];
    opus::static_for<n>([&](auto i) {
        constexpr int io = imm[decltype(i)::value] - imm[0];
        static_assert(io >= 0 && io <= 4095, "LDS issue stride must fit the 12-bit MUBUF offset");
        g.template async_load<VEC>(base, gofs[decltype(i)::value] - io, s_os, opus::number<io>{}, opus::number<AB_AUX>{});
    });
}
// sched_group_barrier masks (LLVM AMDGPU)
constexpr int SCHED_VALU = 0x002;
constexpr int SCHED_MFMA = 0x008;
constexpr int SCHED_VMEM_READ = 0x820;   // 0x020 VMEM read | 0x800 LDS DMA (newer LLVM classifies buffer_load ... lds as LDS DMA)
constexpr int SCHED_DS_READ = 0x100;

template<int REPS, int MFMAS, int OTHER_MASK, int OTHERS, int GROUP = 0>
__device__ inline void sched_pairs() {
    opus::static_for<REPS>([&](auto) {
        __builtin_amdgcn_sched_group_barrier(SCHED_MFMA, MFMAS, GROUP);
        __builtin_amdgcn_sched_group_barrier(OTHER_MASK, OTHERS, GROUP);
    });
}

// One MFMA, then DS reads and VALU.
template<int DS, int VALU>
__device__ inline void sched_m() {
    __builtin_amdgcn_sched_group_barrier(SCHED_MFMA, 1, 0);
    if constexpr (DS > 0) __builtin_amdgcn_sched_group_barrier(SCHED_DS_READ, DS, 0);
    if constexpr (VALU > 0) __builtin_amdgcn_sched_group_barrier(SCHED_VALU, VALU, 0);
}
// C11 (32 MFMAs): 8 DS (A0 of t+1) + 24 VALU (scale transposes of half 0).
__device__ inline void sched_c11() {
    opus::static_for<7>([&](auto) { sched_m<1, 1>(); sched_m<0, 1>(); sched_m<0, 1>(); sched_m<0, 0>(); });
    sched_m<1, 1>();
    __builtin_amdgcn_sched_group_barrier(SCHED_VMEM_READ, 1, 0);   // scale panel LDS-DMA
    sched_m<0, 1>(); sched_m<0, 1>(); sched_m<0, 0>();
}
// C00 (32 MFMAs): 16 DS (A1 / B1 of t+1) in the first 16 MFMAs, so none is left at the end of
// the quadrant, + 24 VALU (scale transposes of half 1).
__device__ inline void sched_c00() {
    opus::static_for<8>([&](auto) { sched_m<1, 1>(); sched_m<1, 1>(); sched_m<0, 1>(); });   // e2a: 16 DS over 24 MFMAs
    opus::static_for<8>([&](auto) { sched_m<0, 0>(); });
}

template<int REPS, int DS, int VMEM, int VALU>
__device__ inline void sched_x4() {
    static_assert(DS <= 2 && VMEM <= 1);
    opus::static_for<REPS>([&](auto) {
        __builtin_amdgcn_sched_group_barrier(SCHED_MFMA, 1, 0);
        if constexpr (DS >= 1) __builtin_amdgcn_sched_group_barrier(SCHED_DS_READ, 1, 0);
        __builtin_amdgcn_sched_group_barrier(SCHED_MFMA, 1, 0);
        if constexpr (VMEM >= 1) __builtin_amdgcn_sched_group_barrier(SCHED_VMEM_READ, 1, 0);
        __builtin_amdgcn_sched_group_barrier(SCHED_MFMA, 1, 0);
        if constexpr (DS >= 2) __builtin_amdgcn_sched_group_barrier(SCHED_DS_READ, 1, 0);
        __builtin_amdgcn_sched_group_barrier(SCHED_MFMA, 1, 0);
        if constexpr (VALU > 0) __builtin_amdgcn_sched_group_barrier(SCHED_VALU, VALU, 0);
    });
}

// B_DIRECT schedules: C01 = 8 DS (scales) + 12 VMEM (4 A DMA + 8 B direct); C00 = 8 DS (A1) + 8 VMEM (B1 direct).
__device__ inline void sched_c01_bd() {
    opus::static_for<8>([&](auto r) {
        __builtin_amdgcn_sched_group_barrier(SCHED_MFMA, 1, 0);
        __builtin_amdgcn_sched_group_barrier(SCHED_DS_READ, 1, 0);
        __builtin_amdgcn_sched_group_barrier(SCHED_MFMA, 1, 0);
        __builtin_amdgcn_sched_group_barrier(SCHED_VMEM_READ, 1, 0);
        __builtin_amdgcn_sched_group_barrier(SCHED_MFMA, 1, 0);
        if constexpr (decltype(r)::value < 4) __builtin_amdgcn_sched_group_barrier(SCHED_VMEM_READ, 1, 0);
        __builtin_amdgcn_sched_group_barrier(SCHED_MFMA, 1, 0);
    });
}
__device__ inline void sched_c00_bd() {
    opus::static_for<8>([&](auto) {
        __builtin_amdgcn_sched_group_barrier(SCHED_MFMA, 1, 0);
        __builtin_amdgcn_sched_group_barrier(SCHED_DS_READ, 1, 0);
        __builtin_amdgcn_sched_group_barrier(SCHED_VALU, 1, 0);
        __builtin_amdgcn_sched_group_barrier(SCHED_MFMA, 1, 0);
        __builtin_amdgcn_sched_group_barrier(SCHED_VMEM_READ, 1, 0);
        __builtin_amdgcn_sched_group_barrier(SCHED_VALU, 1, 0);
        __builtin_amdgcn_sched_group_barrier(SCHED_MFMA, 1, 0);
        __builtin_amdgcn_sched_group_barrier(SCHED_VALU, 1, 0);
    });
    opus::static_for<8>([&](auto) { __builtin_amdgcn_sched_group_barrier(SCHED_MFMA, 1, 0); });
}
__device__ inline void sched_c00_store_bd() {
    opus::static_for<8>([&](auto) {
        __builtin_amdgcn_sched_group_barrier(SCHED_MFMA, 1, 0);
        __builtin_amdgcn_sched_group_barrier(SCHED_DS_READ, 1, 0);
        __builtin_amdgcn_sched_group_barrier(SCHED_MFMA, 1, 0);
        __builtin_amdgcn_sched_group_barrier(0x040, 1, 0);
        __builtin_amdgcn_sched_group_barrier(SCHED_MFMA, 1, 0);
        __builtin_amdgcn_sched_group_barrier(SCHED_VMEM_READ, 1, 0);
        __builtin_amdgcn_sched_group_barrier(SCHED_MFMA, 1, 0);
        __builtin_amdgcn_sched_group_barrier(0x040, 1, 0);
    });
}
#define SCHED_C01() do { sched_x4<4, 2, 1, 0>(); sched_x4<4, 1, 1, 0>(); } while (0)
#define SCHED_C00() sched_c00()
#define SCHED_C00_STORE() sched_c00_store()
#define BB_VMCNT (T::a_buffer_load_insts + T::b_buffer_load_insts)   // C10's A0 + B0 copies
// PF_SPREAD: A1 / B1 copies of tile t+2 moved from C01 into C11 / C00 (TA was saturated in C10 + C01).
__device__ inline void sched_c01_nov() {   // 16 DS, no VMEM
    opus::static_for<8>([&](auto) {
        __builtin_amdgcn_sched_group_barrier(SCHED_MFMA, 1, 0);
        __builtin_amdgcn_sched_group_barrier(SCHED_DS_READ, 1, 0);
        __builtin_amdgcn_sched_group_barrier(SCHED_MFMA, 2, 0);
        __builtin_amdgcn_sched_group_barrier(SCHED_DS_READ, 1, 0);
        __builtin_amdgcn_sched_group_barrier(SCHED_MFMA, 1, 0);
    });
}
__device__ inline void sched_c11_pf() {    // 8 DS (A0) + VALU + 5 VMEM (4 A1 copies + scale panel)
    opus::static_for<8>([&](auto r) {
        __builtin_amdgcn_sched_group_barrier(SCHED_MFMA, 1, 0);
        __builtin_amdgcn_sched_group_barrier(SCHED_DS_READ, 1, 0);
        __builtin_amdgcn_sched_group_barrier(SCHED_VALU, 1, 0);
        __builtin_amdgcn_sched_group_barrier(SCHED_MFMA, 1, 0);
        if constexpr (decltype(r)::value % 2 == 0 || decltype(r)::value == 7) __builtin_amdgcn_sched_group_barrier(SCHED_VMEM_READ, 1, 0);
        __builtin_amdgcn_sched_group_barrier(SCHED_VALU, 1, 0);
        __builtin_amdgcn_sched_group_barrier(SCHED_MFMA, 1, 0);
        __builtin_amdgcn_sched_group_barrier(SCHED_VALU, 1, 0);
        __builtin_amdgcn_sched_group_barrier(SCHED_MFMA, 1, 0);
    });
}
__device__ inline void sched_c00_pf() {    // 16 DS (A1, B1) + VALU + 4 VMEM (B1 copies)
    opus::static_for<8>([&](auto r) {
        __builtin_amdgcn_sched_group_barrier(SCHED_MFMA, 1, 0);
        __builtin_amdgcn_sched_group_barrier(SCHED_DS_READ, 1, 0);
        __builtin_amdgcn_sched_group_barrier(SCHED_VALU, 1, 0);
        __builtin_amdgcn_sched_group_barrier(SCHED_MFMA, 1, 0);
        __builtin_amdgcn_sched_group_barrier(SCHED_DS_READ, 1, 0);
        __builtin_amdgcn_sched_group_barrier(SCHED_VALU, 1, 0);
        __builtin_amdgcn_sched_group_barrier(SCHED_MFMA, 1, 0);
        if constexpr (decltype(r)::value % 2 == 1) __builtin_amdgcn_sched_group_barrier(SCHED_VMEM_READ, 1, 0);
        __builtin_amdgcn_sched_group_barrier(SCHED_VALU, 1, 0);
    });
    opus::static_for<8>([&](auto) { __builtin_amdgcn_sched_group_barrier(SCHED_MFMA, 1, 0); });
}
// Generic quadrant schedule: Q MFMAs with DS reads / VMEM reads / VALU / VMEM writes spread evenly after them.
template<int Q, int DS, int VMEM, int VALU, int STORE = 0>
__device__ inline void sched_spread() {
    opus::static_for<Q>([&](auto i_) {
        constexpr int i = decltype(i_)::value;
        constexpr int d = DS * (i + 1) / Q - DS * i / Q, v = VMEM * (i + 1) / Q - VMEM * i / Q;
        constexpr int u = VALU * (i + 1) / Q - VALU * i / Q, w = STORE * (i + 1) / Q - STORE * i / Q;
        __builtin_amdgcn_sched_group_barrier(SCHED_MFMA, 1, 0);
        if constexpr (d > 0) __builtin_amdgcn_sched_group_barrier(SCHED_DS_READ, d, 0);
        if constexpr (v > 0) __builtin_amdgcn_sched_group_barrier(SCHED_VMEM_READ, v, 0);
        if constexpr (u > 0) __builtin_amdgcn_sched_group_barrier(SCHED_VALU, u, 0);
        if constexpr (w > 0) __builtin_amdgcn_sched_group_barrier(0x040, w, 0);
    });
}
// The hand-tuned schedules are kept for the 256 x 256 tile; other tiles use sched_spread.
#define SCHED(orig, DS, VMEM, VALU, ST) do { if constexpr (T::E_M == 4 && T::E_N == 4) { orig; } \
    else { sched_spread<2 * T::E_M * T::E_N, (DS), (VMEM), (VALU), (ST)>(); } } while (0)
// h1: prologue C00 -- MFMA, DS, MFMA, store, MFMA, DS, MFMA, store.
__device__ inline void sched_c00_store() {
    opus::static_for<8>([&](auto) {
        __builtin_amdgcn_sched_group_barrier(SCHED_MFMA, 1, 0);
        __builtin_amdgcn_sched_group_barrier(SCHED_DS_READ, 1, 0);
        __builtin_amdgcn_sched_group_barrier(SCHED_MFMA, 1, 0);
        __builtin_amdgcn_sched_group_barrier(0x040, 1, 0);   // VMEM write
        __builtin_amdgcn_sched_group_barrier(SCHED_MFMA, 1, 0);
        __builtin_amdgcn_sched_group_barrier(SCHED_DS_READ, 1, 0);
        __builtin_amdgcn_sched_group_barrier(SCHED_MFMA, 1, 0);
        __builtin_amdgcn_sched_group_barrier(0x040, 1, 0);
    });
}
// One fp4 MFMA operand per lane: 32 fp4 = 16 bytes = 4 registers.  Held as 4 x i32 because
// opus::vector_t<fp4_t, 32> is a bit-packed struct, not a register vector (the MFMA builtin
// and the pin attribute need an int vector).
using fp4x32_t = opus::vector_t<opus::i32_t, 4>;

template<class T, int M_REPEAT, int N_REPEAT, int C_REG, bool ZC = false, class MMA, class CF>
__device__ inline void mma_scale_one(
    MMA& mma,
    const fp4x32_t& a,
    const fp4x32_t& b,
    CF& c,
    unsigned int v_sfa,
    unsigned int v_sfb) {
    (void)mma;
    using base_mma = typename opus::remove_cvref_t<MMA>::MMA;
    using i32x8_t = opus::vector_t<opus::i32_t, 8>;
    const i32x8_t a8 = __builtin_shufflevector(a, a, 0, 1, 2, 3, -1, -1, -1, -1);
    const i32x8_t b8 = __builtin_shufflevector(b, b, 0, 1, 2, 3, -1, -1, -1, -1);
    if constexpr (C_REG >= 0) {
        [[clang::amdgpu_pin_agpr(C_REG)]] c = base_mma{}(a8, b8, ZC ? CF{} : c, static_cast<int>(v_sfa), static_cast<int>(v_sfb), opus::number<M_REPEAT>{}, opus::number<N_REPEAT>{});
    } else if constexpr (C_REG <= -2) {
        [[clang::amdgpu_pin_vgpr(-C_REG - 2)]] c = base_mma{}(a8, b8, ZC ? CF{} : c, static_cast<int>(v_sfa), static_cast<int>(v_sfb), opus::number<M_REPEAT>{}, opus::number<N_REPEAT>{});
    } else {
        c = base_mma{}(a8, b8, ZC ? CF{} : c, static_cast<int>(v_sfa), static_cast<int>(v_sfb), opus::number<M_REPEAT>{}, opus::number<N_REPEAT>{});
    }
}

template<class T, int HM, int HN, int M, int N>
constexpr int c_reg() {
    constexpr int frag = (M * T::E_N + N) * 4;
    if constexpr (HN == 0) return 128 + HM * 64 + frag;           // AGPR
    else return -(128 + HM * 64 + frag) - 2;                        // VGPR
}

// ===========================================================================
// Kernel
// ===========================================================================
#ifndef C_AUX
#define C_AUX 2
#endif
#ifndef EPI_VALU
#define EPI_VALU 4
#endif
#ifndef BAR_STEP
#define BAR_STEP 8
#endif
// Kernel body as a device function, so the same code backs the JIT-style template __global__ below and the
// extern "C" entry points of the pre-compiled .co images (gen_co/build_co_gfx950.py).
template<class Traits>
__device__ __forceinline__ void gemm_a4w4_mxfp4_nt_body(opus_gemm_mxfp4_scale_kargs_gfx950 kargs) {
#if defined(__gfx950__)
    using namespace opus;
    using T = opus::remove_cvref_t<Traits>;
    using D_A = opus::fp4_t;    // math types of the MMA
    using D_B = opus::fp4_t;
    using D_ACC = opus::fp32_t;
    using D_C = opus::bf16_t;
    using D_AB = opus::u8_t;
    using D_SF = opus::e8m0_t;  // UE8M0 scale, 1 byte (load<16> = 16 bytes)

    const int num_tiles_n = opus_mxfp4_ceil_div(kargs.n, T::B_N);
    const int wave_id = __builtin_amdgcn_readfirstlane(thread_id_x() / T::WARP_SIZE);
    const int tiles_per_batch = opus_mxfp4_ceil_div(kargs.m, T::B_M) * num_tiles_n;
    const int total_tiles = tiles_per_batch * kargs.batch;
    // MODE 1 / 2: one tile (or tile x K segment) per block; the tile loop below runs once.
    const int split_k = T::SPLITK ? kargs.split_k : 1;
    const int grid_blocks = T::PERSIST ? kargs.grid_blocks : total_tiles;
    const int first_tile = T::SPLITK ? static_cast<int>(block_id_x()) / split_k : static_cast<int>(block_id_x());
    const int k_split = T::SPLITK ? static_cast<int>(block_id_x()) % split_k : 0;   // from the kernel arguments: __builtin_amdgcn_grid_size_x() reads the dispatch packet (host memory)
    // Tile 0 (+ scale panel 0) of this block's next output tile is copied during the last K
    // tile of the current one (see "2. Last tile"); the next prologue then skips it.
    int prefetched = 0;
    // h1: bottom half of the previous tile's C (bf16), read from LDS in its write-back and
    // stored during this tile's prologue C00.
    constexpr int H1_VEC = 16 / sizeof(D_C);
    constexpr int H1_COPIES = T::HALF_B_M * T::B_N / (T::BLOCK_SIZE * H1_VEC);   // 16
    opus::vector_t<D_C, H1_VEC> v_pend_c[H1_COPIES];
    D_C* pend_c_ptr = reinterpret_cast<D_C*>(kargs.ptr_c);
    int pend_c = 0;
    int pend_rows = 0, pend_cols = 0;   // rows / columns of the pending tile inside the matrix (M / N boundary)
    // Out-of-range element offset for masked C stores (byte offset 2^31 > any buffer range we set).
    constexpr int C_OOB = 0x40000000;
    auto store_pend_c = [&](int tid) {
        auto g_cp = make_gmem(pend_c_ptr, pend_c ? static_cast<unsigned int>(pend_rows) * static_cast<unsigned int>(kargs.stride_c) * 2u : 0u);
        if constexpr (T::B_N == 192) {   // 3 column blocks of 64: copy i -> rows (i / 3) * 32 + tid / 8, columns (i % 3) * 64 + (tid % 8) * 8
            opus::static_for<H1_COPIES>([&](auto i) {
                constexpr int I = decltype(i)::value;
                const int r = (I / 3) * 32 + tid / 8 + T::HALF_B_M, c = (I % 3) * 64 + (tid % 8) * 8;
                store<H1_VEC>(g_cp, v_pend_c[I], r * kargs.stride_c + c + (c < pend_cols ? 0 : C_OOB), 0, opus::number<C_AUX>{});
            });
        } else {
        // A thread's column is the same in every copy (copies step by whole rows): one out-of-range add per thread.
        const int oob = (tid * H1_VEC) % T::B_N < pend_cols ? 0 : C_OOB;
        opus::static_for<H1_COPIES>([&](auto i) {
            const int linear = tid * H1_VEC + decltype(i)::value * T::BLOCK_SIZE * H1_VEC;
            const int r = linear / T::B_N + T::HALF_B_M, c = linear % T::B_N;
            store<H1_VEC>(g_cp, v_pend_c[i.value], r * kargs.stride_c + c + oob, 0, opus::number<C_AUX>{});
        });
        }
    };
    opus::vector_t<opus::e8m0_t, Traits::VEC_GLOBAL_SCALE> v_nsfa0, v_nsfb0;   // next tile's scale panel 0
    // A / B operands live across output tiles (B_DIRECT loads the next tile's B0 in the last K tile).
    fp4x32_t v_a[2][T::E_K][T::E_M];   // [half][kk]: E_M fp4 operands, 16 B each
    fp4x32_t v_b[2][T::E_K][T::E_N];
#pragma clang loop unroll(disable)
    for (int tile_id = first_tile; tile_id < total_tiles; tile_id += grid_blocks) {
    const int batch_id = tile_id / tiles_per_batch;
    const int tile_in_batch = tile_id % tiles_per_batch;
    const int block_m = tile_in_batch / num_tiles_n;
    const int block_n = tile_in_batch % num_tiles_n;
    // Split-K: this block's K segment [k0, k0 + k / S); bases below are moved to k0 (A: bytes, B: 2 KB per K tile in each
    // 16-row block of shuffle_weight(16,16), scales: 256 B per K tile in each 32-row block of e8m0_shuffle).
    const int k_seg = T::SPLITK ? kargs.k / split_k : kargs.k;
    const int kt0 = T::SPLITK ? k_split * (k_seg / T::B_K) : 0;     // first K tile
    const int loops = k_seg / T::B_K;
    const int row = block_m * T::B_M;
    const int col = block_n * T::B_N;
    // lane id from mbcnt over an opaque all-ones mask: the compiler still knows it is in
    // [0, 64) but cannot hoist the lane-dependent offsets out of the tile loop (kept live
    // through the write-back they spill, and the reloads put vmcnt(0) into the prologue).
    unsigned int lane_mask = ~0u;
    asm volatile("" : "+s"(lane_mask));
    const int lane_id = static_cast<int>(__builtin_amdgcn_mbcnt_hi(lane_mask, __builtin_amdgcn_mbcnt_lo(lane_mask, 0u)));
    const int wave_id_m = wave_id % T::T_M;
    const int wave_id_n = wave_id / T::T_M;

    // M / N boundary: rows of A / B (and C) past the matrix are outside the buffer range -- loads return 0,
    // stores are dropped; C columns past N are masked per 16 B copy (see copy_output_half).
    const int rows_in = T::BOUNDARY ? kargs.m - row : T::B_M, cols_in = T::BOUNDARY ? kargs.n - col : T::B_N;
    const int a_k0 = kt0 * T::B_K_BYTES, b_k0 = kt0 * T::B_K_BYTES * T::W_N, sf_k0 = kt0 * 256;
    auto g_a = make_gmem(reinterpret_cast<const D_AB*>(kargs.ptr_a) + batch_id * kargs.stride_a_batch + row * kargs.stride_a + a_k0,
                         T::BOUNDARY ? static_cast<unsigned int>(rows_in) * static_cast<unsigned int>(kargs.stride_a) - a_k0 : 0xffffffffu);
    auto g_b = make_gmem(reinterpret_cast<const D_AB*>(kargs.ptr_b) + batch_id * kargs.stride_b_batch + col * kargs.stride_b + b_k0,
                         T::BOUNDARY ? static_cast<unsigned int>(cols_in) * static_cast<unsigned int>(kargs.stride_b) - b_k0 : 0xffffffffu);
    auto g_c = make_gmem(reinterpret_cast<D_C*>(kargs.ptr_c) + batch_id * kargs.stride_c_batch + row * kargs.stride_c + col,
                         T::BOUNDARY ? static_cast<unsigned int>(rows_in) * static_cast<unsigned int>(kargs.stride_c) * 2u : 0xffffffffu);
    auto g_sfa = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfa) + batch_id * kargs.stride_sfa_batch + (row / 32) * kargs.stride_sfa + sf_k0, static_cast<unsigned int>(opus_mxfp4_ceil_div(kargs.m - row, 32) * kargs.stride_sfa - sf_k0));   // sc: e8m0_shuffle layout
    auto g_sfb = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfb) + batch_id * kargs.stride_sfb_batch + (col / 32) * kargs.stride_sfb + sf_k0, static_cast<unsigned int>(opus_mxfp4_ceil_div(kargs.n - col, 32) * kargs.stride_sfb - sf_k0));

    const auto u_ga = make_layout_ga<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_a);
    const auto u_gb = make_layout_gb<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_b);
    const auto u_sa = make_layout_sa<T>(wave_id_m, wave_id_n);
    const auto u_sb = make_layout_sb<T>(wave_id_m, wave_id_n);
    const auto u_ra = make_layout_ra<T>(lane_id, wave_id_m);
    const auto u_rb = make_layout_rb<T>(lane_id, wave_id_n);

    constexpr int smem_a_elem = T::smem_m_rep * (T::smem_linear_wave_a + T::smem_padding); 
    constexpr int smem_b_elem = T::smem_n_rep * (T::smem_linear_wave_b + T::smem_padding);
    constexpr int smem_a_bytes = smem_a_elem * 4 * sizeof(D_A);
    constexpr int smem_b_bytes = smem_b_elem * 4 * sizeof(D_B);
    __shared__ __attribute__((aligned(16))) char smem_ab[smem_a_bytes + smem_b_bytes];
    // Stage-major LDS: [A stage 0][B stage 0][A stage 1][B stage 1].  Stage 1 of A and B is
    // then one contiguous 66 KB block, which stages half of C at the original row stride while
    // stage 0 receives the next tile's K tile 0.
    char* const smem_a = smem_ab;
    char* const smem_b = smem_ab + smem_a_bytes / 2;
    __shared__ char smem_sfa[T::SCALE_PANEL_BUFFERS * T::SFA_PANEL_BYTES * sizeof(D_SF)];
    __shared__ char smem_sfb[T::SCALE_PANEL_BUFFERS * T::SFB_PANEL_BYTES * sizeof(D_SF)];
    auto s_a = make_smem(reinterpret_cast<D_AB*>(smem_a));
    auto s_b = make_smem(reinterpret_cast<D_AB*>(smem_b));
    auto s_sfa = make_smem(reinterpret_cast<D_SF*>(smem_sfa));
    auto s_sfb = make_smem(reinterpret_cast<D_SF*>(smem_sfb));

    auto mma = make_tiled_mma<D_A, D_B, D_ACC>(
        seq<T::E_M, T::E_N, 1>{},          // K: one MFMA (W_K = 128); the E_K = 2 steps are v_a[.][kk]
        seq<T::T_M, T::T_N, T::T_K>{},
        seq<T::W_M, T::W_N, T::W_K>{},
        mfma_adaptor_swap_ab{});
    using MMA = decltype(mma);
    using CFrag = opus::vector_t<D_ACC, MMA::mma_c_len>;   // one 16x16 C fragment, 4 fp32 per lane
    CFrag v_c[2][2][T::E_M * T::E_N];
    opus::static_for<T::E_M * T::E_N>([&](auto f) { clear(v_c[0][0][f.value]); });

    using D_SF_PACK = opus::u32_t;
    D_SF_PACK v_ssa[2][T::E_M], v_ssb[2][T::E_N];
#define SFA(h, kk, m) v_ssa[h][m]
#define SFB(h, kk, n) v_ssb[h][n]
#define SOA(kk, m) (2 * (kk))
#define SOB(kk, n) (2 * (kk))
    D_SF_PACK v_sfa[2][T::E_K];
    D_SF_PACK v_sfb[2][T::E_K];

    // ---- offsets ----
    auto ga_offset = [&](int half_tile_m, int tile_k) { return half_tile_m * T::HALF_B_M * kargs.stride_a + tile_k * T::B_K_BYTES; };
    auto gb_offset = [&](int half_tile_n, int tile_k) { return half_tile_n * T::HALF_B_N * kargs.stride_b + tile_k * (T::B_K_BYTES * T::W_N); };   // bs: K tile = 2 KB per 16-row block
    constexpr int stage_elems = 2 * smem_a_elem + 2 * smem_b_elem;   // bytes (D_AB = u8)
    auto sa_offset = [&](int stage, int half_tile_m) { return stage * stage_elems + half_tile_m * smem_a_elem; };
    auto sb_offset = [&](int stage, int half_tile_n) { return stage * stage_elems + half_tile_n * smem_b_elem; };
    // sc: e8m0_shuffle layout = [32-row block][K tile (8 groups)][k_lane 4][row 16][kk 2][row16-pair 2], 256 B per
    // (block, K tile).  A panel (2 K tiles x 256 rows) is 8 runs of 512 contiguous bytes: wave w copies blocks 2w, 2w+1.
    static_assert(T::SCALE_PANEL_TILES == 2 && T::NUM_KGROUPS == 8 && T::SF_WAVES_A <= T::NUM_WAVES && T::SF_WAVES_B <= T::NUM_WAVES);
    const int wave_sfa = wave_id % T::SF_WAVES_A, wave_sfb = wave_id % T::SF_WAVES_B;   // extra waves repeat a copy
    const int sfa_g_lane = (2 * wave_sfa + lane_id / 32) * kargs.stride_sfa + (lane_id % 32) * T::VEC_GLOBAL_SCALE;
    const int sfb_g_lane = (2 * wave_sfb + lane_id / 32) * kargs.stride_sfb + (lane_id % 32) * T::VEC_GLOBAL_SCALE;
    auto gsfa_offset = [&](int panel_k_begin) { return sfa_g_lane + panel_k_begin * 256; };
    auto gsfb_offset = [&](int panel_k_begin) { return sfb_g_lane + panel_k_begin * 256; };
    auto ssfa_offset = [&](int stage, int tile_in_panel) { return stage * T::SFA_PANEL_BYTES + tile_in_panel * T::NUM_KGROUPS; };
    auto ssfb_offset = [&](int stage, int tile_in_panel) { return stage * T::SFB_PANEL_BYTES + tile_in_panel * T::NUM_KGROUPS; };

    // ---- global -> LDS ----
    // Odd LDS-DMA issue counts (E = 3): wave w copies chunks w * NI + i (same chunk -> rows map as the layouts:
    // A chunk c = rows (c / 2) * 16 + 2 * (lane / 8) + c % 2, B chunk c = 16-row block c / 2, K half c % 2), so a chunk
    // pair can straddle two waves and the per-issue parity is a (wave-uniform) runtime value.
    constexpr int DMA_CHUNK = T::smem_linear_wave_a + T::smem_padding;   // 1088
    const int a_kcol = ((lane_id % 8) ^ ((((lane_id / 8) >> 1) & 1) << 1)) * T::VEC_A;
    auto dma_odd_a = [&](auto& g, int lds_off, int s_os) {
        constexpr int NI = T::a_buffer_load_insts;
        const int w = wave_id_n * 2 + wave_id_m;
        char* const base = reinterpret_cast<char*>(s_a.ptr) + lds_off + w * NI * DMA_CHUNK;
        opus::static_for<NI>([&](auto i_) {
            constexpr int i = decltype(i_)::value, io = i * DMA_CHUNK;
            const int c = w * NI + i;
            const int goff = ((c >> 1) * 16 + 2 * (lane_id / 8) + (c & 1)) * kargs.stride_a + a_kcol;
            g.template async_load<T::VEC_A>(base, goff - io, s_os, opus::number<io>{}, opus::number<AB_AUX>{});
        });
    };
    auto dma_odd_b = [&](auto& g, int lds_off, int s_os) {
        constexpr int NI = T::b_buffer_load_insts;
        const int w = wave_id_m * 2 + wave_id_n;
        char* const base = reinterpret_cast<char*>(s_b.ptr) + lds_off + w * NI * DMA_CHUNK;
        opus::static_for<NI>([&](auto i_) {
            constexpr int i = decltype(i_)::value, io = i * DMA_CHUNK;
            const int c = w * NI + i;
            const int goff = (c >> 1) * T::W_N * kargs.stride_b + (c & 1) * 1024 + lane_id * T::VEC_B;
            g.template async_load<T::VEC_B>(base, goff - io, s_os, opus::number<io>{}, opus::number<AB_AUX>{});
        });
    };
    auto prefetch_a = [&](int stage, int tile, int half_tile_m) {
        if constexpr (T::A_ODD) dma_odd_a(g_a, sa_offset(stage, half_tile_m), ga_offset(half_tile_m, tile));
        else async_load_imm<T::VEC_A>(g_a, s_a.ptr, u_ga, u_sa + sa_offset(stage, half_tile_m), ga_offset(half_tile_m, tile));
    };
    auto prefetch_b = [&](int stage, int tile, int half_tile_n) {
        if constexpr (T::B_ODD) dma_odd_b(g_b, sb_offset(stage, half_tile_n), gb_offset(half_tile_n, tile));
        else async_load_imm<T::VEC_B>(g_b, s_b.ptr, u_gb, u_sb + sb_offset(stage, half_tile_n), gb_offset(half_tile_n, tile));
    };
    // Scale panels go global -> VGPR -> 4x4 byte transpose -> LDS, so LDS holds the MFMA scale
    // layout and the main loop reads it as is (one transpose per panel instead of one per wave
    // and tile).  A panel is loaded in iteration t and written in t + 1 (its load has landed by
    // B_b of t + 1).
    using SF_PANEL = opus::vector_t<D_SF, T::VEC_GLOBAL_SCALE>;
    using SF_DW4 = opus::vector_t<opus::u32_t, 4>;
    SF_PANEL v_sf_pend;
    auto fetch_sfa = [&](int panel_k_begin) { return load<T::VEC_GLOBAL_SCALE>(g_sfa, gsfa_offset(panel_k_begin)); };
    auto fetch_sfb = [&](int panel_k_begin) { return load<T::VEC_GLOBAL_SCALE>(g_sfb, gsfb_offset(panel_k_begin)); };
    // 4x4 byte transpose inside each lane quad (lane = 4 r + e, byte g = K group), as PR 6696:
    //   step 1, partner e ^ 1: even e -> [own0, nb0, own2, nb2], odd e -> [nb1, own1, nb3, own3]
    //   step 2, partner e ^ 2: e < 2 -> [own0, own1, nb0, nb1], e >= 2 -> [nb2, nb3, own2, own3]
    // v_perm(S0 = neighbour, S1 = own): selector bytes 0-3 pick S1, 4-7 pick S0.
    const unsigned int sf_q = static_cast<unsigned int>(lane_id % 4);
    const unsigned int sf_sel1 = (sf_q & 1) ? 0x03070105u : 0x06020400u;
    const unsigned int sf_sel2 = (sf_q & 2) ? 0x03020706u : 0x05040100u;
    auto quad_transpose = [&](unsigned int x) {
        const unsigned int y = static_cast<unsigned int>(__builtin_amdgcn_mov_dpp(static_cast<int>(x), 0xB1, 0xF, 0xF, true));   // quad_perm [1,0,3,2]
        const unsigned int t = __builtin_amdgcn_perm(y, x, sf_sel1);
        const unsigned int z = static_cast<unsigned int>(__builtin_amdgcn_mov_dpp(static_cast<int>(t), 0x4E, 0xF, 0xF, true));   // quad_perm [2,3,0,1]
        return __builtin_amdgcn_perm(z, t, sf_sel2);
    };
    // Lane 4 r + g writes the 16 B its reader lane g * 16 + r reads.
    // w2 sf: panel image [tile_in_panel][row][8 B] (2 x 2048 B per buffer), rows XOR-swizzled
    // within a wave's 64 rows: slot s -> s ^ (((s >> 4) & 3) << 2).  The writer's 8-lane /
    // 16-lane groups and the reader's 32-lane groups are then bank-conflict free (microbenchmark).
    const int sf_put_off = ((lane_id % 4) * 16 + ((lane_id / 4) ^ ((lane_id % 4) << 2))) * T::VEC_LDS_SCALE;
    auto put_sf = [&](auto& s_sf_, int wave_base, const SF_PANEL& raw, int buf) {   // sc: no transpose, contiguous 1 KB per wave
        store<T::VEC_GLOBAL_SCALE>(s_sf_, raw, buf * T::SFA_PANEL_BYTES + wave_base);
    };
    const int ssfa_wbase = wave_id * (T::WARP_SIZE * T::VEC_GLOBAL_SCALE) + lane_id * T::VEC_GLOBAL_SCALE;   // sc
    const int ssfb_wbase = ssfa_wbase;
    auto dma_sfa = [&](int panel_k_begin, int buf) {
        g_sfa.template async_load<T::VEC_GLOBAL_SCALE>(reinterpret_cast<char*>(s_sfa.ptr) + buf * T::SFA_PANEL_BYTES + wave_sfa * (T::WARP_SIZE * T::VEC_GLOBAL_SCALE), gsfa_offset(panel_k_begin));
    };
    auto dma_sfb = [&](int panel_k_begin, int buf) {
        g_sfb.template async_load<T::VEC_GLOBAL_SCALE>(reinterpret_cast<char*>(s_sfb.ptr) + buf * T::SFB_PANEL_BYTES + wave_sfb * (T::WARP_SIZE * T::VEC_GLOBAL_SCALE), gsfb_offset(panel_k_begin));
    };
#define SF_STEP_EVEN() dma_sfa(tile + 2, sf_wr)
#define SF_STEP_ODD()  dma_sfb(tile + 3, sf_wr)

    // ---- LDS -> VGPR ----
    const auto ra_offs = opus::layout_to_offsets<T::VEC_A>(u_ra);   // [repeat]
    const auto rb_offs = opus::layout_to_offsets<T::VEC_B>(u_rb);
    auto load_a = [&](auto& sa_, auto half_) {
        constexpr int half = decltype(half_)::value;
        opus::static_for<T::E_K>([&](auto kk) { opus::static_for<T::E_M>([&](auto e) {
            constexpr int agpr = 0 + half * 32 + (kk.value * T::E_M + e.value) * 4;
            [[clang::amdgpu_pin_agpr(agpr)]] v_a[half][kk.value][e.value] =
                __builtin_bit_cast(fp4x32_t, load<T::VEC_A>(sa_, ra_offs[e.value] + sa_offset(0, half) + kk.value * T::W_K_BYTES));
        }); });
    };
    auto load_b = [&](auto& sb_, auto half_) {
        constexpr int half = decltype(half_)::value;
        opus::static_for<T::E_K>([&](auto kk) { opus::static_for<T::E_N>([&](auto e) {
            constexpr int agpr = 64 + half * 32 + (kk.value * T::E_N + e.value) * 4;
            [[clang::amdgpu_pin_agpr(agpr)]] v_b[half][kk.value][e.value] =
                __builtin_bit_cast(fp4x32_t, load<T::VEC_B>(sb_, rb_offs[e.value] + sb_offset(0, half) + kk.value * (T::smem_linear_wave_b + T::smem_padding)));   // bs: kk = next chunk
        }); });
    };
#define LOAD_B(rd, h, kt) load_b(rd, opus::number<h>{})
    // Scale reads: lane reads the 2 dwords (kk = 0, 1) of half h at LDS index (h * 2 + w) * 64 +
    // lane (+ 8 B for the second tile of the panel), early (C01) into v_sf*_next; read_sf moves
    // half h into v_sfa / v_sfb when the previous values are no longer used.
    using SF_KK = opus::vector_t<D_SF_PACK, T::E_K>;
    // sc: each dword D_e (row block h*4+e) holds bytes [kk0 w0, kk0 w1, kk1 w0, kk1 w1] of lane (g, r); the wave
    // needs w = its wave_m (A) / wave_n (B).  Two v_perm stages build [D0, D1, D2, D3] at byte w (kk0) / 2 + w (kk1).
    const unsigned int sfa_sel = 0x06020400u + static_cast<unsigned int>(wave_id_m) * 0x01010101u;
    const unsigned int sfb_sel = 0x06020400u + static_cast<unsigned int>(wave_id_n) * 0x01010101u;
    const int sf_rd_base = lane_id * 4;
    const unsigned int sfa_sh = 8u * static_cast<unsigned int>(wave_id_m), sfb_sh = 8u * static_cast<unsigned int>(wave_id_n);
    (void)sfa_sh; (void)sfb_sh;
    // sc2: load_sf* only issues the LDS reads (raw dwords, C01); the v_perm gather runs in read_sf (C11 / C00),
    // long after the data landed, so no lgkmcnt wait lands in the main loop.
    using SF_RAW_A = opus::vector_t<opus::u32_t, T::E_M>;
    using SF_RAW_B = opus::vector_t<opus::u32_t, T::E_N>;
    SF_RAW_A v_sfa_raw[2];
    SF_RAW_B v_sfb_raw[2];
    int sf_read_off = 0;
    // Half h, repeat e = 32-row block h * E + e of the panel (512 B per block: 2 K tiles x 256 B).
    int sf_read_off_b = 0;
    auto sf_load_raw = [&](auto& rd_, auto h_, auto e_cnt_, int read_off) {
        constexpr int h = decltype(h_)::value, E = decltype(e_cnt_)::value;
        opus::vector_t<opus::u32_t, E> r;
        opus::static_for<E>([&](auto e) {
            r[e.value] = __builtin_bit_cast(unsigned int, load<4>(rd_, sf_rd_base + (h * E + e.value) * 512 + read_off));
        });
        return r;
    };
    auto sf_gather = [&](const SF_RAW_A& dw, unsigned int sel) {
        const unsigned int t01 = __builtin_amdgcn_perm(dw[1], dw[0], sel);
        const unsigned int t23 = __builtin_amdgcn_perm(dw[3], dw[2], sel);
        return SF_KK{__builtin_amdgcn_perm(t23, t01, 0x05040100u), __builtin_amdgcn_perm(t23, t01, 0x07060302u)};
    };
    auto load_sfa = [&](auto& rd_sfa_, int buf, int tile_in_panel) {
        sf_read_off = buf * T::SFA_PANEL_BYTES + tile_in_panel * 256;
        sf_read_off_b = buf * T::SFB_PANEL_BYTES + tile_in_panel * 256;
        opus::static_for<2>([&](auto h) { v_sfa_raw[h.value] = sf_load_raw(rd_sfa_, h, opus::number<T::E_M>{}, sf_read_off); });
    };
    auto load_sfb = [&](auto& rd_sfb_, int, int) {
        opus::static_for<2>([&](auto h) { v_sfb_raw[h.value] = sf_load_raw(rd_sfb_, h, opus::number<T::E_N>{}, sf_read_off_b); });
    };

    //
    // Before (load<8> x 2 halves = 16 e8m0, row-major; lane 16g+r holds row g*32+r and row 128+g*32+r):
    //            dword0 [0][0]        dword1 [0][1]        dword2 [1][0]                dword3 [1][1]
    // lane 0     M0k0  M0k1  M0k2  M0k3   M0k4  M0k5  M0k6  M0k7   M128k0 M128k1 M128k2 M128k3   M128k4 M128k5 M128k6 M128k7
    // lane 16    M32k0 M32k1 M32k2 M32k3  M32k4 M32k5 M32k6 M32k7  M160k0 M160k1 M160k2 M160k3   M160k4 M160k5 M160k6 M160k7
    // lane 32    M64k0 M64k1 M64k2 M64k3  M64k4 M64k5 M64k6 M64k7  M192k0 M192k1 M192k2 M192k3   M192k4 M192k5 M192k6 M192k7
    // lane 48    M96k0 M96k1 M96k2 M96k3  M96k4 M96k5 M96k6 M96k7  M224k0 M224k1 M224k2 M224k3   M224k4 M224k5 M224k6 M224k7
    //
    // After (byte e = repeat e, picked by op_sel; lane group g = K group g of the MFMA):
    //            v_sfa[0][0]           v_sfa[0][1]           v_sfa[1][0]                 v_sfa[1][1]
    // lane 0     M0k0 M32k0 M64k0 M96k0  M0k4 M32k4 M64k4 M96k4  M128k0 M160k0 M192k0 M224k0  M128k4 M160k4 M192k4 M224k4
    // lane 16    M0k1 M32k1 M64k1 M96k1  M0k5 M32k5 M64k5 M96k5  M128k1 M160k1 M192k1 M224k1  M128k5 M160k5 M192k5 M224k5
    // lane 32    M0k2 M32k2 M64k2 M96k2  M0k6 M32k6 M64k6 M96k6  M128k2 M160k2 M192k2 M224k2  M128k6 M160k6 M192k6 M224k6
    // lane 48    M0k3 M32k3 M64k3 M96k3  M0k7 M32k7 M64k7 M96k7  M128k3 M160k3 M192k3 M224k3  M128k7 M160k7 M192k7 M224k7
    //
    auto read_sf = [&](auto&, auto&, auto h_) {
        constexpr int h = decltype(h_)::value;
        auto shr2 = [](unsigned int lo, unsigned int hi, unsigned int sh, unsigned int& o0, unsigned int& o1) {
            const unsigned long long v = ((static_cast<unsigned long long>(hi) << 32) | lo) >> sh;
            o0 = static_cast<unsigned int>(v); o1 = static_cast<unsigned int>(v >> 32);
        };
        opus::static_for<T::E_M / 2>([&](auto q) { shr2(v_sfa_raw[h][2 * q.value], v_sfa_raw[h][2 * q.value + 1], sfa_sh, v_ssa[h][2 * q.value], v_ssa[h][2 * q.value + 1]); });
        opus::static_for<T::E_N / 2>([&](auto q) { shr2(v_sfb_raw[h][2 * q.value], v_sfb_raw[h][2 * q.value + 1], sfb_sh, v_ssb[h][2 * q.value], v_ssb[h][2 * q.value + 1]); });
        if constexpr (T::E_M % 2) v_ssa[h][T::E_M - 1] = v_sfa_raw[h][T::E_M - 1] >> sfa_sh;   // odd E: last dword alone
        if constexpr (T::E_N % 2) v_ssb[h][T::E_N - 1] = v_sfb_raw[h][T::E_N - 1] >> sfb_sh;
        (void)sf_gather;
    };

    // LDS reads go through __restrict__ copies of the views, passed as function parameters
    // (clang ignores __restrict__ on locals): without them the compiler cannot tell the
    // reads from the in-flight LDS-DMA copies and puts a vmcnt(0) before the reads.
    auto views = [&](OPUS_LDS_ADDR char* __restrict__ read_a, OPUS_LDS_ADDR char* __restrict__ read_b,
                     OPUS_LDS_ADDR char* __restrict__ read_sfa, OPUS_LDS_ADDR char* __restrict__ read_sfb, auto&& body) {
        auto rd_a = s_a;     rd_a.ptr = read_a;
        auto rd_b = s_b;     rd_b.ptr = read_b;
        auto rd_sfa = s_sfa; rd_sfa.ptr = read_sfa;
        auto rd_sfb = s_sfb; rd_sfb.ptr = read_sfb;
        body(rd_a, rd_b, rd_sfa, rd_sfb);
    };

    // ===== Prologue =====
    constexpr int frags = T::E_M * T::E_N;   // 16 (m, n) steps per quadrant, 2 MFMAs each
    // Per-quadrant instruction counts for the generic schedules (per wave).
    constexpr int QA = T::a_buffer_load_insts, QB = T::b_buffer_load_insts;   // A / B LDS-DMA issues per half tile
    constexpr int RA = T::E_K * T::E_M, RB = T::E_K * T::E_N;                  // A / B ds_read_b128 per half tile
    constexpr int SFR = T::E_M + T::E_N;                                       // scale ds_read2st64 per K tile
    constexpr int SV = (T::E_M + 1) / 2 + (T::E_N + 1) / 2;                    // scale shifts per half
    // Issue tile 0 and scale panel 0, then tile 1 and panel 1 (tile 2, 3); wait only for
    // tile 0 + panel 0 (counted vmcnt), read A0 / B0 and the scales of tile 0, and compute
    // C00 of tile 0 while A1 / B1 of tile 0 are read.  The main loop then starts with
    // "C00 of tile t is done".
    // SF_DMA2: panel 0 (sfa, sfb) and sfb of panel 1 go global -> LDS by DMA (for a prefetched
    // tile already issued in the previous tile's C11), so no VGPR scale load is pending here
    // and the compiler inserts no vmcnt(0).  The counted wait lets the previous tile's 16
    // top-half C stores (issued after those copies) and tile 1 stay in flight.
    if (!prefetched) {
        prefetch_a(0, 0, 0);
        prefetch_b(0, 0, 0);
        prefetch_a(0, 0, 1);
        prefetch_b(0, 0, 1);
        dma_sfa(0, 0);
        dma_sfb(0, 0);
        dma_sfb(T::SCALE_PANEL_TILES, 1);
    }
    if (loops > 1) {
        prefetch_a(1, 1, 0);
        prefetch_b(1, 1, 0);
        prefetch_a(1, 1, 1);
        prefetch_b(1, 1, 1);
        if (prefetched) s_waitcnt_vmcnt(number<2 * (T::a_buffer_load_insts + T::b_buffer_load_insts) + T::C_COPIES>{});   // tile 1 + C stores may fly
        else            s_waitcnt_vmcnt(number<2 * (T::a_buffer_load_insts + T::b_buffer_load_insts)>{});        // tile 1 may fly
    } else {
        s_waitcnt_vmcnt(0_I);
    }
    views(s_a.ptr, s_b.ptr, s_sfa.ptr, s_sfb.ptr, [&](auto& rd_a, auto& rd_b, auto& rd_sfa, auto& rd_sfb) {
        // Tile 0 (copied by every wave) is visible after this barrier: A0 / B0 reads go out first and
        // their latency is covered by the scale transposes / writes, which need a second barrier.
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
        load_a(rd_a, number<0>{});
        load_b(rd_b, number<0>{});
        __builtin_amdgcn_sched_group_barrier(SCHED_DS_READ, T::E_K * (T::E_M + T::E_N), 0);   // A0, B0 first
        __builtin_amdgcn_sched_barrier(0);
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
        load_sfa(rd_sfa, 0, 0);
        load_sfb(rd_sfb, 0, 0);
        read_sf(rd_sfa, rd_sfb, number<0>{});
        read_sf(rd_sfa, rd_sfb, number<1>{});
        __builtin_amdgcn_sched_barrier(0);
        // C00 of tile 0 + A1, B1 of tile 0 (+ h1: the previous tile's bottom-half stores).
        load_a(rd_a, number<1>{});
        LOAD_B(rd_b, 1, 0);
        if constexpr (T::PERSIST) store_pend_c(wave_id * T::WARP_SIZE + lane_id);
        opus::static_for<frags>([&](auto i) {
            constexpr int m = decltype(i)::value / T::E_N, n = decltype(i)::value % T::E_N;
            mma_scale_one<T, SOA(0, m), SOB(0, n), c_reg<T, 0, 0, m, n>()>(mma, v_a[0][0][m], v_b[0][0][n], v_c[0][0][m * T::E_N + n], SFA(0, 0, m), SFB(0, 0, n));   // kk = 0
            mma_scale_one<T, SOA(1, m), SOB(1, n), c_reg<T, 0, 0, m, n>()>(mma, v_a[0][1][m], v_b[0][1][n], v_c[0][0][m * T::E_N + n], SFA(0, 1, m), SFB(0, 1, n));   // kk = 1
        });
        SCHED(SCHED_C00_STORE(), RA + RB, 0, 0, H1_COPIES);   // h1
        __builtin_amdgcn_sched_barrier(0);
    });

    // ===== Main loop =====
    int tile = 0;
    // Scale panel buffers tracked instead of (k / 2) % 3: sf_rd holds the panel of tile t+1,
    // sf_wr = sf_rd + 1 (mod 3) receives the panel copied in iteration t (t+3 for odd t,
    // t+2 for even t -- both one panel after tile t+1's); after an even t, sf_rd advances.
    int sf_rd = 0;
    // Two tiles per trip: stage / next_stage and the panel parity are compile-time in each
    // copy; A / B / C are pinned, so both copies use the same registers.
    // ZC_PEEL: the first trip is a separate instance whose first MFMA of C10 / C01 / C11 (kk = 0) uses
    // C = 0, so those accumulators need no clearing at the start of the tile.
    auto main_trip = [&](auto first_) {
        constexpr bool ZFIRST = decltype(first_)::value;
        {   // even tile
            constexpr int stage = 0;
            constexpr int next_stage = 1;
            const int sf_wr = sf_rd == T::SCALE_PANEL_BUFFERS - 1 ? 0 : sf_rd + 1;
            views(s_a.ptr + sa_offset(next_stage, 0), s_b.ptr + sb_offset(next_stage, 0), s_sfa.ptr, s_sfb.ptr,
                  [&](auto& rd_a, auto& rd_b, auto& rd_sfa, auto& rd_sfb) {
                // B_a: every wave is done reading stage(t) (A1 / B1 of tile t in C00), so
                // tile t+2 may be copied into it.
                s_waitcnt_lgkmcnt(0_I);
                __builtin_amdgcn_s_barrier();
                __builtin_amdgcn_sched_barrier(0);

                // C10(t) + A0, B0 of tile t+2 -> stage(t), one buffer_load per 4 MFMAs.
                prefetch_a(stage, tile + 2, 0);
                prefetch_b(stage, tile + 2, 0);
                opus::static_for<frags>([&](auto i) {
                    constexpr int m = decltype(i)::value / T::E_N, n = decltype(i)::value % T::E_N;
                    mma_scale_one<T, SOA(0, m), SOB(0, n), c_reg<T, 1, 0, m, n>(), ZFIRST>(mma, v_a[1][0][m], v_b[0][0][n], v_c[1][0][m * T::E_N + n], SFA(1, 0, m), SFB(0, 0, n));   // kk = 0
                    mma_scale_one<T, SOA(1, m), SOB(1, n), c_reg<T, 1, 0, m, n>()>(mma, v_a[1][1][m], v_b[0][1][n], v_c[1][0][m * T::E_N + n], SFA(1, 1, m), SFB(0, 1, n));   // kk = 1
                });
                SCHED((sched_x4<8, 0, 1, 0>()), 0, QA + QB, 0, 0);
                __builtin_amdgcn_sched_barrier(0);
                // B_b: tile t+1 (and its scale panel) landed; the 8 copies just issued may still fly.
                s_waitcnt_vmcnt(number<BB_VMCNT>{});
                __builtin_amdgcn_s_barrier();
                __builtin_amdgcn_sched_barrier(0);

                // C01(t) + scales and B0 of tile t+1 + A1, B1 of tile t+2.
                load_sfa(rd_sfa, sf_rd, 1);
                load_sfb(rd_sfb, sf_rd, 1);
                LOAD_B(rd_b, 0, tile + 1);
                opus::static_for<frags>([&](auto i) {
                    constexpr int m = decltype(i)::value / T::E_N, n = decltype(i)::value % T::E_N;
                    mma_scale_one<T, SOA(0, m), SOB(0, n), c_reg<T, 0, 1, m, n>(), ZFIRST>(mma, v_a[0][0][m], v_b[1][0][n], v_c[0][1][m * T::E_N + n], SFA(0, 0, m), SFB(1, 0, n));   // kk = 0
                    mma_scale_one<T, SOA(1, m), SOB(1, n), c_reg<T, 0, 1, m, n>()>(mma, v_a[0][1][m], v_b[1][1][n], v_c[0][1][m * T::E_N + n], SFA(0, 1, m), SFB(1, 1, n));   // kk = 1
                });
                SCHED(sched_c01_nov(), RB + SFR, 0, 0, 0);
                __builtin_amdgcn_sched_barrier(0);

                // Scale panel copies, split over two tiles (one panel instruction touches 64 cache
                // lines, ~5x an A/B copy on the TA): odd t -> sfa, even t -> sfb of the panel
                // starting at t + 3 / t + 2.  With 3 panel buffers it overwrites the panel three
                // back (read long ago) and lands before the B_b that reads it.  A runtime branch,
                // so it sits on a region boundary.
                // The two branches end in different (empty) asm, so the compiler cannot merge
                // their tails into one copy that selects between the sfa / sfb layouts (that
                // selection goes through scratch, whose vmcnt wait drains every copy in flight).
                __builtin_amdgcn_sched_barrier(0);

                // C11(t) + A0 of tile t+1 + scale transposes of half 0 of tile t+1.
                load_a(rd_a, number<0>{});
                read_sf(rd_sfa, rd_sfb, number<0>{});
                prefetch_a(stage, tile + 2, 1);
                SF_STEP_EVEN();   // even tile: sfa of the panel starting at tile + 2 (+ sfb of tile + 4 pending)
                opus::static_for<frags>([&](auto i) {
                    constexpr int m = decltype(i)::value / T::E_N, n = decltype(i)::value % T::E_N;
                    mma_scale_one<T, SOA(0, m), SOB(0, n), c_reg<T, 1, 1, m, n>(), ZFIRST>(mma, v_a[1][0][m], v_b[1][0][n], v_c[1][1][m * T::E_N + n], SFA(1, 0, m), SFB(1, 0, n));   // kk = 0
                    mma_scale_one<T, SOA(1, m), SOB(1, n), c_reg<T, 1, 1, m, n>()>(mma, v_a[1][1][m], v_b[1][1][n], v_c[1][1][m * T::E_N + n], SFA(1, 1, m), SFB(1, 1, n));   // kk = 1
                });
                SCHED(sched_c11_pf(), RA, QA + 1, SV, 0);
                __builtin_amdgcn_sched_barrier(0);

                // C00(t+1) + A1, B1 of tile t+1 + scale transposes of half 1 of tile t+1.
                load_a(rd_a, number<1>{});
                LOAD_B(rd_b, 1, tile + 1);
                read_sf(rd_sfa, rd_sfb, number<1>{});
                prefetch_b(stage, tile + 2, 1);
                opus::static_for<frags>([&](auto i) {
                    constexpr int m = decltype(i)::value / T::E_N, n = decltype(i)::value % T::E_N;
                    mma_scale_one<T, SOA(0, m), SOB(0, n), c_reg<T, 0, 0, m, n>()>(mma, v_a[0][0][m], v_b[0][0][n], v_c[0][0][m * T::E_N + n], SFA(0, 0, m), SFB(0, 0, n));   // kk = 0
                    mma_scale_one<T, SOA(1, m), SOB(1, n), c_reg<T, 0, 0, m, n>()>(mma, v_a[0][1][m], v_b[0][1][n], v_c[0][0][m * T::E_N + n], SFA(0, 1, m), SFB(0, 1, n));   // kk = 1
                });
                SCHED(sched_c00_pf(), RA + RB, QB, SV, 0);
                __builtin_amdgcn_sched_barrier(0);
            });
            sf_rd = sf_wr;                                     // advances after an even tile
        }
        {   // odd tile
            const int tile_odd = tile + 1;
            const int tile = tile_odd;
            constexpr int stage = 1;
            constexpr int next_stage = 0;
            const int sf_wr = sf_rd == T::SCALE_PANEL_BUFFERS - 1 ? 0 : sf_rd + 1;
            views(s_a.ptr + sa_offset(next_stage, 0), s_b.ptr + sb_offset(next_stage, 0), s_sfa.ptr, s_sfb.ptr,
                  [&](auto& rd_a, auto& rd_b, auto& rd_sfa, auto& rd_sfb) {
                // B_a: every wave is done reading stage(t) (A1 / B1 of tile t in C00), so
                // tile t+2 may be copied into it.
                s_waitcnt_lgkmcnt(0_I);
                __builtin_amdgcn_s_barrier();
                __builtin_amdgcn_sched_barrier(0);

                // C10(t) + A0, B0 of tile t+2 -> stage(t), one buffer_load per 4 MFMAs.
                prefetch_a(stage, tile + 2, 0);
                prefetch_b(stage, tile + 2, 0);
                opus::static_for<frags>([&](auto i) {
                    constexpr int m = decltype(i)::value / T::E_N, n = decltype(i)::value % T::E_N;
                    mma_scale_one<T, SOA(0, m), SOB(0, n), c_reg<T, 1, 0, m, n>()>(mma, v_a[1][0][m], v_b[0][0][n], v_c[1][0][m * T::E_N + n], SFA(1, 0, m), SFB(0, 0, n));   // kk = 0
                    mma_scale_one<T, SOA(1, m), SOB(1, n), c_reg<T, 1, 0, m, n>()>(mma, v_a[1][1][m], v_b[0][1][n], v_c[1][0][m * T::E_N + n], SFA(1, 1, m), SFB(0, 1, n));   // kk = 1
                });
                SCHED((sched_x4<8, 0, 1, 0>()), 0, QA + QB, 0, 0);
                __builtin_amdgcn_sched_barrier(0);
                // B_b: tile t+1 (and its scale panel) landed; the 8 copies just issued may still fly.
                s_waitcnt_vmcnt(number<BB_VMCNT>{});
                __builtin_amdgcn_s_barrier();
                __builtin_amdgcn_sched_barrier(0);

                // C01(t) + scales and B0 of tile t+1 + A1, B1 of tile t+2.
                load_sfa(rd_sfa, sf_rd, 0);
                load_sfb(rd_sfb, sf_rd, 0);
                LOAD_B(rd_b, 0, tile + 1);
                opus::static_for<frags>([&](auto i) {
                    constexpr int m = decltype(i)::value / T::E_N, n = decltype(i)::value % T::E_N;
                    mma_scale_one<T, SOA(0, m), SOB(0, n), c_reg<T, 0, 1, m, n>()>(mma, v_a[0][0][m], v_b[1][0][n], v_c[0][1][m * T::E_N + n], SFA(0, 0, m), SFB(1, 0, n));   // kk = 0
                    mma_scale_one<T, SOA(1, m), SOB(1, n), c_reg<T, 0, 1, m, n>()>(mma, v_a[0][1][m], v_b[1][1][n], v_c[0][1][m * T::E_N + n], SFA(0, 1, m), SFB(1, 1, n));   // kk = 1
                });
                SCHED(sched_c01_nov(), RB + SFR, 0, 0, 0);
                __builtin_amdgcn_sched_barrier(0);

                // Scale panel copies, split over two tiles (one panel instruction touches 64 cache
                // lines, ~5x an A/B copy on the TA): odd t -> sfa, even t -> sfb of the panel
                // starting at t + 3 / t + 2.  With 3 panel buffers it overwrites the panel three
                // back (read long ago) and lands before the B_b that reads it.  A runtime branch,
                // so it sits on a region boundary.
                // The two branches end in different (empty) asm, so the compiler cannot merge
                // their tails into one copy that selects between the sfa / sfb layouts (that
                // selection goes through scratch, whose vmcnt wait drains every copy in flight).
                // odd tile: sfa of the panel starting at tile + 3; past the last panel it only
                // overwrites the buffer of a consumed panel (harmless), so no branch.
                __builtin_amdgcn_sched_barrier(0);

                // C11(t) + A0 of tile t+1 + scale transposes of half 0 of tile t+1.
                load_a(rd_a, number<0>{});
                read_sf(rd_sfa, rd_sfb, number<0>{});
                prefetch_a(stage, tile + 2, 1);
                SF_STEP_ODD();    // odd tile: sfb of the panel starting at tile + 3
                opus::static_for<frags>([&](auto i) {
                    constexpr int m = decltype(i)::value / T::E_N, n = decltype(i)::value % T::E_N;
                    mma_scale_one<T, SOA(0, m), SOB(0, n), c_reg<T, 1, 1, m, n>()>(mma, v_a[1][0][m], v_b[1][0][n], v_c[1][1][m * T::E_N + n], SFA(1, 0, m), SFB(1, 0, n));   // kk = 0
                    mma_scale_one<T, SOA(1, m), SOB(1, n), c_reg<T, 1, 1, m, n>()>(mma, v_a[1][1][m], v_b[1][1][n], v_c[1][1][m * T::E_N + n], SFA(1, 1, m), SFB(1, 1, n));   // kk = 1
                });
                SCHED(sched_c11_pf(), RA, QA + 1, SV, 0);
                __builtin_amdgcn_sched_barrier(0);

                // C00(t+1) + A1, B1 of tile t+1 + scale transposes of half 1 of tile t+1.
                load_a(rd_a, number<1>{});
                LOAD_B(rd_b, 1, tile + 1);
                read_sf(rd_sfa, rd_sfb, number<1>{});
                prefetch_b(stage, tile + 2, 1);
                opus::static_for<frags>([&](auto i) {
                    constexpr int m = decltype(i)::value / T::E_N, n = decltype(i)::value % T::E_N;
                    mma_scale_one<T, SOA(0, m), SOB(0, n), c_reg<T, 0, 0, m, n>()>(mma, v_a[0][0][m], v_b[0][0][n], v_c[0][0][m * T::E_N + n], SFA(0, 0, m), SFB(0, 0, n));   // kk = 0
                    mma_scale_one<T, SOA(1, m), SOB(1, n), c_reg<T, 0, 0, m, n>()>(mma, v_a[0][1][m], v_b[0][1][n], v_c[0][0][m * T::E_N + n], SFA(0, 1, m), SFB(0, 1, n));   // kk = 1
                });
                SCHED(sched_c00_pf(), RA + RB, QB, SV, 0);
                __builtin_amdgcn_sched_barrier(0);
            });
        }
        };
    if (tile + 3 < loops) { main_trip(opus::bool_constant<true>{}); tile += 2; }
    else {
        opus::static_for<2>([&](auto hm) { opus::static_for<2>([&](auto hn) { opus::static_for<T::E_M * T::E_N>([&](auto f) {
            if constexpr (hm.value + hn.value > 0) clear(v_c[hm.value][hn.value][f.value]);
        }); }); });
    }
#pragma clang loop unroll(disable)
    for (; tile + 3 < loops; tile += 2) main_trip(opus::bool_constant<false>{});
    // At most one more steady iteration (odd number of them).
#pragma clang loop unroll(disable)
    for (; tile + 2 < loops; ++tile) {
        const int stage = tile & 1;
        const int next_stage = stage ^ 1;
        const int sf_wr = sf_rd == T::SCALE_PANEL_BUFFERS - 1 ? 0 : sf_rd + 1;
        views(s_a.ptr + sa_offset(next_stage, 0), s_b.ptr + sb_offset(next_stage, 0), s_sfa.ptr, s_sfb.ptr,
              [&](auto& rd_a, auto& rd_b, auto& rd_sfa, auto& rd_sfb) {
            // B_a: every wave is done reading stage(t) (A1 / B1 of tile t in C00), so
            // tile t+2 may be copied into it.
            s_waitcnt_lgkmcnt(0_I);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            // C10(t) + A0, B0 of tile t+2 -> stage(t), one buffer_load per 4 MFMAs.
            prefetch_a(stage, tile + 2, 0);
            prefetch_b(stage, tile + 2, 0);
            opus::static_for<frags>([&](auto i) {
                constexpr int m = decltype(i)::value / T::E_N, n = decltype(i)::value % T::E_N;
                mma_scale_one<T, SOA(0, m), SOB(0, n), c_reg<T, 1, 0, m, n>()>(mma, v_a[1][0][m], v_b[0][0][n], v_c[1][0][m * T::E_N + n], SFA(1, 0, m), SFB(0, 0, n));   // kk = 0
                mma_scale_one<T, SOA(1, m), SOB(1, n), c_reg<T, 1, 0, m, n>()>(mma, v_a[1][1][m], v_b[0][1][n], v_c[1][0][m * T::E_N + n], SFA(1, 1, m), SFB(0, 1, n));   // kk = 1
            });
            SCHED((sched_x4<8, 0, 1, 0>()), 0, QA + QB, 0, 0);
            __builtin_amdgcn_sched_barrier(0);
            // B_b: tile t+1 (and its scale panel) landed; the 8 copies just issued may still fly.
            s_waitcnt_vmcnt(number<BB_VMCNT>{});
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            // C01(t) + scales and B0 of tile t+1 + A1, B1 of tile t+2.
            load_sfa(rd_sfa, sf_rd, (tile + 1) & 1);
            load_sfb(rd_sfb, sf_rd, (tile + 1) & 1);
            LOAD_B(rd_b, 0, tile + 1);
            prefetch_a(stage, tile + 2, 1);
            prefetch_b(stage, tile + 2, 1);
            opus::static_for<frags>([&](auto i) {
                constexpr int m = decltype(i)::value / T::E_N, n = decltype(i)::value % T::E_N;
                mma_scale_one<T, SOA(0, m), SOB(0, n), c_reg<T, 0, 1, m, n>()>(mma, v_a[0][0][m], v_b[1][0][n], v_c[0][1][m * T::E_N + n], SFA(0, 0, m), SFB(1, 0, n));   // kk = 0
                mma_scale_one<T, SOA(1, m), SOB(1, n), c_reg<T, 0, 1, m, n>()>(mma, v_a[0][1][m], v_b[1][1][n], v_c[0][1][m * T::E_N + n], SFA(0, 1, m), SFB(1, 1, n));   // kk = 1
            });
            SCHED(SCHED_C01(), RB + SFR, QA + QB, 0, 0);
            __builtin_amdgcn_sched_barrier(0);

            // Scale panel copies, split over two tiles (one panel instruction touches 64 cache
            // lines, ~5x an A/B copy on the TA): odd t -> sfa, even t -> sfb of the panel
            // starting at t + 3 / t + 2.  With 3 panel buffers it overwrites the panel three
            // back (read long ago) and lands before the B_b that reads it.  A runtime branch,
            // so it sits on a region boundary.
            // The two branches end in different (empty) asm, so the compiler cannot merge
            // their tails into one copy that selects between the sfa / sfb layouts (that
            // selection goes through scratch, whose vmcnt wait drains every copy in flight).
            if (tile & 1) {
                SF_STEP_ODD();
                asm volatile("; panel copy: sfa" ::: "memory");
            } else {
                SF_STEP_EVEN();
                asm volatile("; panel copy: sfb" ::: "memory");
            }
            __builtin_amdgcn_sched_barrier(0);

            // C11(t) + A0 of tile t+1 + scale transposes of half 0 of tile t+1.
            load_a(rd_a, number<0>{});
            read_sf(rd_sfa, rd_sfb, number<0>{});
            opus::static_for<frags>([&](auto i) {
                constexpr int m = decltype(i)::value / T::E_N, n = decltype(i)::value % T::E_N;
                mma_scale_one<T, SOA(0, m), SOB(0, n), c_reg<T, 1, 1, m, n>()>(mma, v_a[1][0][m], v_b[1][0][n], v_c[1][1][m * T::E_N + n], SFA(1, 0, m), SFB(1, 0, n));   // kk = 0
                mma_scale_one<T, SOA(1, m), SOB(1, n), c_reg<T, 1, 1, m, n>()>(mma, v_a[1][1][m], v_b[1][1][n], v_c[1][1][m * T::E_N + n], SFA(1, 1, m), SFB(1, 1, n));   // kk = 1
            });
            SCHED(sched_c11(), RA, 1, SV, 0);
            __builtin_amdgcn_sched_barrier(0);

            // C00(t+1) + A1, B1 of tile t+1 + scale transposes of half 1 of tile t+1.
            load_a(rd_a, number<1>{});
            LOAD_B(rd_b, 1, tile + 1);
            read_sf(rd_sfa, rd_sfb, number<1>{});
            opus::static_for<frags>([&](auto i) {
                constexpr int m = decltype(i)::value / T::E_N, n = decltype(i)::value % T::E_N;
                mma_scale_one<T, SOA(0, m), SOB(0, n), c_reg<T, 0, 0, m, n>()>(mma, v_a[0][0][m], v_b[0][0][n], v_c[0][0][m * T::E_N + n], SFA(0, 0, m), SFB(0, 0, n));   // kk = 0
                mma_scale_one<T, SOA(1, m), SOB(1, n), c_reg<T, 0, 0, m, n>()>(mma, v_a[0][1][m], v_b[0][1][n], v_c[0][0][m * T::E_N + n], SFA(0, 1, m), SFB(0, 1, n));   // kk = 1
            });
            SCHED(SCHED_C00(), RA + RB, 0, SV, 0);
            __builtin_amdgcn_sched_barrier(0);
        });
        sf_rd = (tile & 1) ? sf_rd : sf_wr;
    }

    // ===== Epilogue =====
    // 1. Second-to-last tile: C10, C01, C11 of it and C00 of the last tile; no more copies.
    if (tile + 1 < loops) {
        const int next_stage = (tile & 1) ^ 1;
        views(s_a.ptr + sa_offset(next_stage, 0), s_b.ptr + sb_offset(next_stage, 0), s_sfa.ptr, s_sfb.ptr,
              [&](auto& rd_a, auto& rd_b, auto& rd_sfa, auto& rd_sfb) {
            // C10(t).
            opus::static_for<frags>([&](auto i) {
                constexpr int m = decltype(i)::value / T::E_N, n = decltype(i)::value % T::E_N;
                mma_scale_one<T, SOA(0, m), SOB(0, n), c_reg<T, 1, 0, m, n>()>(mma, v_a[1][0][m], v_b[0][0][n], v_c[1][0][m * T::E_N + n], SFA(1, 0, m), SFB(0, 0, n));   // kk = 0
                mma_scale_one<T, SOA(1, m), SOB(1, n), c_reg<T, 1, 0, m, n>()>(mma, v_a[1][1][m], v_b[0][1][n], v_c[1][0][m * T::E_N + n], SFA(1, 1, m), SFB(0, 1, n));   // kk = 1
            });
            SCHED((sched_x4<8, 0, 0, 0>()), 0, 0, 0, 0);
            __builtin_amdgcn_sched_barrier(0);
            // B_b: tile t+1 (and its scale panel) landed.
            s_waitcnt_vmcnt(0_I);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            // C01(t) + scales and B0 of tile t+1.
            load_sfa(rd_sfa, sf_rd, (tile + 1) & 1);
            load_sfb(rd_sfb, sf_rd, (tile + 1) & 1);
            LOAD_B(rd_b, 0, tile + 1);
            opus::static_for<frags>([&](auto i) {
                constexpr int m = decltype(i)::value / T::E_N, n = decltype(i)::value % T::E_N;
                mma_scale_one<T, SOA(0, m), SOB(0, n), c_reg<T, 0, 1, m, n>()>(mma, v_a[0][0][m], v_b[1][0][n], v_c[0][1][m * T::E_N + n], SFA(0, 0, m), SFB(1, 0, n));   // kk = 0
                mma_scale_one<T, SOA(1, m), SOB(1, n), c_reg<T, 0, 1, m, n>()>(mma, v_a[0][1][m], v_b[1][1][n], v_c[0][1][m * T::E_N + n], SFA(0, 1, m), SFB(1, 1, n));   // kk = 1
            });
            SCHED((sched_x4<4, 2, 0, 0>(), sched_x4<4, 1, 0, 0>()), RB + SFR, 0, 0, 0);
            __builtin_amdgcn_sched_barrier(0);

            // C11(t) + A0 of tile t+1 + scale transposes of half 0 of tile t+1.
            load_a(rd_a, number<0>{});
            read_sf(rd_sfa, rd_sfb, number<0>{});
            opus::static_for<frags>([&](auto i) {
                constexpr int m = decltype(i)::value / T::E_N, n = decltype(i)::value % T::E_N;
                mma_scale_one<T, SOA(0, m), SOB(0, n), c_reg<T, 1, 1, m, n>()>(mma, v_a[1][0][m], v_b[1][0][n], v_c[1][1][m * T::E_N + n], SFA(1, 0, m), SFB(1, 0, n));   // kk = 0
                mma_scale_one<T, SOA(1, m), SOB(1, n), c_reg<T, 1, 1, m, n>()>(mma, v_a[1][1][m], v_b[1][1][n], v_c[1][1][m * T::E_N + n], SFA(1, 1, m), SFB(1, 1, n));   // kk = 1
            });
            SCHED(sched_c11(), RA, 1, SV, 0);
            __builtin_amdgcn_sched_barrier(0);

            // C00(t+1) + A1, B1 of tile t+1 + scale transposes of half 1 of tile t+1.
            load_a(rd_a, number<1>{});
            LOAD_B(rd_b, 1, tile + 1);
            read_sf(rd_sfa, rd_sfb, number<1>{});
            opus::static_for<frags>([&](auto i) {
                constexpr int m = decltype(i)::value / T::E_N, n = decltype(i)::value % T::E_N;
                mma_scale_one<T, SOA(0, m), SOB(0, n), c_reg<T, 0, 0, m, n>()>(mma, v_a[0][0][m], v_b[0][0][n], v_c[0][0][m * T::E_N + n], SFA(0, 0, m), SFB(0, 0, n));   // kk = 0
                mma_scale_one<T, SOA(1, m), SOB(1, n), c_reg<T, 0, 0, m, n>()>(mma, v_a[0][1][m], v_b[0][1][n], v_c[0][0][m * T::E_N + n], SFA(0, 1, m), SFB(0, 1, n));   // kk = 1
            });
            SCHED(SCHED_C00(), RA + RB, 0, SV, 0);
            __builtin_amdgcn_sched_barrier(0);
        });
        ++tile;
    }
    // (write-back helpers, defined before the last tile so EPI_OVL can stage C inside it)
    constexpr int c_lds_stride = T::B_N + 8;
    constexpr int VEC_OUT = 16 / sizeof(D_C);                                            // 8 bf16 = 16 B
    static_assert(T::HALF_B_M * c_lds_stride * sizeof(D_C) <= stage_elems * sizeof(D_AB));   // s1: stage is 2 KB larger
    auto s_c = make_smem(reinterpret_cast<D_C*>(smem_ab + stage_elems));
    // Re-derive the thread position here (opaque) so the write-back offsets are computed now
    // instead of being kept live (and spilled) from the prologue.
    // (from the SGPR wave id and mbcnt, so thread_id_x() / v0 need not stay live)
    unsigned int lane_mask_e = ~0u;
    asm volatile("" : "+s"(lane_mask_e));
    const int lane_e = static_cast<int>(__builtin_amdgcn_mbcnt_hi(lane_mask_e, __builtin_amdgcn_mbcnt_lo(lane_mask_e, 0u)));
    const int wave_e = wave_id;
    const int tid_e = wave_e * T::WARP_SIZE + lane_e;
    auto p_coord_c = opus::make_tuple(wave_e % T::T_M, lane_e % mma.grpn_c, wave_e / T::T_M, lane_e / mma.grpn_c);
    auto u_sc = partition_layout_c<T::VEC_C>(mma, opus::make_tuple(opus::number<c_lds_stride>{}, 1_I), p_coord_c);
    const int output_thread_id = tid_e;
    // w2 c: rows with bit 3 set store their 8 B (4 bf16) pieces with the two halves of each
    // 16 B chunk swapped, so the 16-lane groups of ds_write_b64 hit distinct banks (the
    // 528 B row stride maps rows r and r + 8 to the same bank).  Read row = tid / 32 + 8 i,
    // so its bit 3 is i & 1 (compile time): odd reads swap the halves back in registers.
    const int c_swz = T::C_SWZ ? ((lane_e >> 3) & 1) * (4 - 8 * ((lane_e / mma.grpn_c) & 1)) : 0;
    auto c_unswz = [](auto v, auto i_) {
        if constexpr (!T::C_SWZ || (decltype(i_)::value & 1) == 0) return v;
        else {
            using U4 = opus::vector_t<opus::u32_t, 4>;
            const U4 u = __builtin_bit_cast(U4, v);
            return __builtin_bit_cast(decltype(v), U4{u[2], u[3], u[0], u[1]});
        }
    };
    // bf16 C: LDS -> global, one half tile.  All LDS reads are issued before the stores: the
    // compiler does not move a ds_read above a buffer_store (it cannot tell them apart), so a
    // read / store loop would expose every read latency.
    constexpr int c_copies = T::HALF_B_M * T::B_N / (T::BLOCK_SIZE * VEC_OUT);   // 16
    static_assert(T::B_N == 192 || T::BLOCK_SIZE * 8 % T::B_N == 0, "a thread's C column is the same in every copy");
    static_assert(T::B_N != 192 || (c_copies % 3 == 0 && T::HALF_B_M % 32 == 0));
    // B_N = 192: copy i of thread t -> staged row (i / 3) * 32 + t / 8, column (i % 3) * 64 + (t % 8) * 8.
    auto c_row = [&](int t, auto i_) { constexpr int I = decltype(i_)::value; return (I / 3) * 32 + t / 8; };
    auto c_col = [&](int t, auto i_) { constexpr int I = decltype(i_)::value; return (I % 3) * 64 + (t % 8) * 8; };
    auto copy_output_half = [&](int half_tile_m) {
      if constexpr (T::B_N == 192) {
        opus::vector_t<D_C, VEC_OUT> v[c_copies];
        opus::static_for<c_copies>([&](auto i) {
            v[i.value] = load<VEC_OUT>(s_c, c_row(output_thread_id, i) * c_lds_stride + c_col(output_thread_id, i));
        });
        opus::static_for<c_copies>([&](auto i) {
            const int r = c_row(output_thread_id, i) + half_tile_m * T::HALF_B_M, c = c_col(output_thread_id, i);
            store<VEC_OUT>(g_c, v[i.value], r * kargs.stride_c + c + (c < cols_in ? 0 : C_OOB), 0, opus::number<C_AUX>{});
        });
      } else {
        const int oob = (output_thread_id * VEC_OUT) % T::B_N < cols_in ? 0 : C_OOB;
        opus::vector_t<D_C, VEC_OUT> v[c_copies];
        opus::static_for<c_copies>([&](auto i) {
            const int linear = output_thread_id * VEC_OUT + decltype(i)::value * T::BLOCK_SIZE * VEC_OUT;
            v[i.value] = c_unswz(load<VEC_OUT>(s_c, (linear / T::B_N) * c_lds_stride + linear % T::B_N), i);
        });
        opus::static_for<c_copies>([&](auto i) {
            const int linear = output_thread_id * VEC_OUT + decltype(i)::value * T::BLOCK_SIZE * VEC_OUT;
            const int r = linear / T::B_N + half_tile_m * T::HALF_B_M, c = linear % T::B_N;
            store<VEC_OUT>(g_c, v[i.value], r * kargs.stride_c + c + oob, 0, opus::number<C_AUX>{});
        });
      }
    };
    // fp32 accumulators -> bf16 -> LDS (columns hn * 128 of the staged half).
    auto stage_quadrant = [&](const auto& acc, int half_tile_n) {
        typename MMA::vtype_c quad;
        opus::static_for<T::E_M * T::E_N>([&](auto f) {
            opus::set_slice(quad, acc[f.value], opus::number<f.value * MMA::mma_c_len>{}, opus::number<(f.value + 1) * MMA::mma_c_len>{});
        });
        store<T::VEC_C>(s_c, cast<D_C>(quad), u_sc + (half_tile_n * T::HALF_B_N + c_swz));
    };
    auto sync_lds = [&]() {
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
    };
    // 2. Last tile: C10, C01, C11.  Every LDS read of this output tile is done (after the
    //    barrier), so K tile 0 and scale panel 0 of the block's next output tile are copied
    //    into stage 0 / panel buffer 0 now, one copy per 4 MFMAs.  Without a next tile the
    //    copies use an empty buffer range (no memory traffic, they only zero stage 0, which is
    //    not read again) so the interleaving needs no branch.
    {
        const int next_id = tile_id + grid_blocks;
        const bool has_next = next_id < total_tiles;
        const int nid = has_next ? next_id : tile_id;
        const int n_batch = nid / tiles_per_batch;
        const int n_in_batch = nid % tiles_per_batch;
        const int n_row = (n_in_batch / num_tiles_n) * T::B_M;
        const int n_col = (n_in_batch % num_tiles_n) * T::B_N;
        auto ng_a = make_gmem(reinterpret_cast<const D_AB*>(kargs.ptr_a) + n_batch * kargs.stride_a_batch + n_row * kargs.stride_a,
                              has_next ? static_cast<unsigned int>(kargs.m - n_row) * static_cast<unsigned int>(kargs.stride_a) : 0u);
        auto ng_b = make_gmem(reinterpret_cast<const D_AB*>(kargs.ptr_b) + n_batch * kargs.stride_b_batch + n_col * kargs.stride_b,
                              has_next ? static_cast<unsigned int>(kargs.n - n_col) * static_cast<unsigned int>(kargs.stride_b) : 0u);
        auto ng_sfa = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfa) + n_batch * kargs.stride_sfa_batch + (n_row / 32) * kargs.stride_sfa, has_next ? static_cast<unsigned int>(opus_mxfp4_ceil_div(kargs.m - n_row, 32) * kargs.stride_sfa) : 0u);
        auto ng_sfb = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfb) + n_batch * kargs.stride_sfb_batch + (n_col / 32) * kargs.stride_sfb, has_next ? static_cast<unsigned int>(opus_mxfp4_ceil_div(kargs.n - n_col, 32) * kargs.stride_sfb) : 0u);
        prefetched = T::PERSIST ? has_next : false;

        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // C10(t) + A0, B0 of the next tile's K tile 0.
        if constexpr (T::PERSIST) {
        if constexpr (T::A_ODD) dma_odd_a(ng_a, sa_offset(0, 0), ga_offset(0, 0));
        else async_load_imm<T::VEC_A>(ng_a, s_a.ptr, u_ga, u_sa + sa_offset(0, 0), ga_offset(0, 0));
        if constexpr (T::B_ODD) dma_odd_b(ng_b, sb_offset(0, 0), gb_offset(0, 0));
        else async_load_imm<T::VEC_B>(ng_b, s_b.ptr, u_gb, u_sb + sb_offset(0, 0), gb_offset(0, 0));
        }
        opus::static_for<frags>([&](auto i) {
            constexpr int m = decltype(i)::value / T::E_N, n = decltype(i)::value % T::E_N;
            mma_scale_one<T, SOA(0, m), SOB(0, n), c_reg<T, 1, 0, m, n>()>(mma, v_a[1][0][m], v_b[0][0][n], v_c[1][0][m * T::E_N + n], SFA(1, 0, m), SFB(0, 0, n));   // kk = 0
            mma_scale_one<T, SOA(1, m), SOB(1, n), c_reg<T, 1, 0, m, n>()>(mma, v_a[1][1][m], v_b[0][1][n], v_c[1][0][m * T::E_N + n], SFA(1, 1, m), SFB(0, 1, n));   // kk = 1
        });
        SCHED((sched_x4<8, 0, 1, 0>()), 0, QA + QB, 0, 0);
        __builtin_amdgcn_sched_barrier(0);

        // C01(t) + A1, B1 of the next tile's K tile 0.
        if constexpr (T::PERSIST) {
        if constexpr (T::A_ODD) dma_odd_a(ng_a, sa_offset(0, 1), ga_offset(1, 0));
        else async_load_imm<T::VEC_A>(ng_a, s_a.ptr, u_ga, u_sa + sa_offset(0, 1), ga_offset(1, 0));
        if constexpr (T::B_ODD) dma_odd_b(ng_b, sb_offset(0, 1), gb_offset(1, 0));
        else async_load_imm<T::VEC_B>(ng_b, s_b.ptr, u_gb, u_sb + sb_offset(0, 1), gb_offset(1, 0));
        }
        opus::static_for<frags>([&](auto i) {
            constexpr int m = decltype(i)::value / T::E_N, n = decltype(i)::value % T::E_N;
            mma_scale_one<T, SOA(0, m), SOB(0, n), c_reg<T, 0, 1, m, n>()>(mma, v_a[0][0][m], v_b[1][0][n], v_c[0][1][m * T::E_N + n], SFA(0, 0, m), SFB(1, 0, n));   // kk = 0
            mma_scale_one<T, SOA(1, m), SOB(1, n), c_reg<T, 0, 1, m, n>()>(mma, v_a[0][1][m], v_b[1][1][n], v_c[0][1][m * T::E_N + n], SFA(0, 1, m), SFB(1, 1, n));   // kk = 1
        });
        SCHED((sched_x4<8, 0, 1, 0>()), 0, QA + QB, 0, 0);
        __builtin_amdgcn_sched_barrier(0);

        // C11(t) + scale panel 0 of the next tile.
        if constexpr (T::PERSIST) {   // sc4: lane offsets re-derived here (opaque lane id) so the per-tile ones need not stay live / spill
            unsigned int lm = ~0u; asm volatile("" : "+s"(lm));
            const int ln = static_cast<int>(__builtin_amdgcn_mbcnt_hi(lm, __builtin_amdgcn_mbcnt_lo(lm, 0u)));
            const int ws_a = wave_id % T::SF_WAVES_A, ws_b = wave_id % T::SF_WAVES_B;
            const int blk = 2 * ws_a + ln / 32, blk_b = 2 * ws_b + ln / 32, in_blk = (ln % 32) * T::VEC_GLOBAL_SCALE;
            char* const sfa_dst = reinterpret_cast<char*>(s_sfa.ptr) + ws_a * (T::WARP_SIZE * T::VEC_GLOBAL_SCALE);
            char* const sfb_dst = reinterpret_cast<char*>(s_sfb.ptr) + ws_b * (T::WARP_SIZE * T::VEC_GLOBAL_SCALE);
            ng_sfa.template async_load<T::VEC_GLOBAL_SCALE>(sfa_dst, blk * kargs.stride_sfa + in_blk);
            ng_sfb.template async_load<T::VEC_GLOBAL_SCALE>(sfb_dst, blk_b * kargs.stride_sfb + in_blk);
            ng_sfb.template async_load<T::VEC_GLOBAL_SCALE>(sfb_dst + T::SFB_PANEL_BYTES, blk_b * kargs.stride_sfb + in_blk + T::SCALE_PANEL_TILES * 256);
        }
        opus::static_for<frags>([&](auto i) {
            constexpr int m = decltype(i)::value / T::E_N, n = decltype(i)::value % T::E_N;
            mma_scale_one<T, SOA(0, m), SOB(0, n), c_reg<T, 1, 1, m, n>()>(mma, v_a[1][0][m], v_b[1][0][n], v_c[1][1][m * T::E_N + n], SFA(1, 0, m), SFB(1, 0, n));   // kk = 0
            mma_scale_one<T, SOA(1, m), SOB(1, n), c_reg<T, 1, 1, m, n>()>(mma, v_a[1][1][m], v_b[1][1][n], v_c[1][1][m * T::E_N + n], SFA(1, 1, m), SFB(1, 1, n));   // kk = 1
        });
        SCHED((sched_x4<2, 0, 1, 0>(), sched_x4<6, 0, 0, 0>()), 0, 3, 0, 0);
        __builtin_amdgcn_sched_barrier(0);
    }

    // 3. Write back C (bf16) through LDS, as opus_gemm_pipeline_a8w8_blockscale_mxscale_
    //    bpreshuffle_4wave_pin_agpr_gfx950 (PR 6696, Output::lds).  A wave owns only 16
    //    columns of a row per fragment (the other wave_n holds the neighbours), so direct
    //    fragment stores leave every 128-byte line partially written.  The whole 256 x 256
    //    tile is staged as one bf16 image in the idle A+B LDS, rows padded to B_N + 8
    //    elements (the 16 rows of a fragment then step by 16 bytes across the banks);
    //    each quadrant gets its own region, so staging needs no buffer reuse.  Per
    //    quadrant: stage -> sync -> 8 copies of 16 B per thread (one buffer_store = 4
    //    rows x 256 B), non-temporal (aux 2).
    // Staging uses only stage 1 (stage 0 receives the next tile's K tile 0): half of C
    // (128 rows x 256 columns) at a time, rows padded to B_N + 8 elements (528 B, as before),
    // which is exactly the 66 KB of stage 1.  Top half (Q00 / Q01), then bottom half.
    // No vmcnt waits here: the next tile's K tile 0 copies are in flight.  Stage 1 was last
    // read before the barrier at the start of "2. Last tile".
    if constexpr (T::SPLITK) {
        // Split-K: S > 1 -> store this segment's fp32 partial tile (C fragment layout, row stride B_N) to the workspace;
        // the last block to arrive for this tile (atomic counter, reset by that block) adds the other S - 1 partials to
        // its accumulators and writes C below.  Agent-scope fences: the blocks of a tile may sit on different XCDs.
        // Broadcast flag in the scale-panel LDS (no longer read after the last K tile; LDS is full for 256 x 256).
        int* const sk_last_p = reinterpret_cast<int*>(smem_sfa);
        if (split_k > 1) {
            const long tile_elems = static_cast<long>(T::B_M) * T::B_N;
            float* const ws_tiles = reinterpret_cast<float*>(reinterpret_cast<char*>(kargs.ptr_ws) + opus_mxfp4_sk_counter_bytes(total_tiles));
            int* const counters = reinterpret_cast<int*>(kargs.ptr_ws);
            auto u_wc = partition_layout_c<T::VEC_C>(mma, opus::make_tuple(opus::number<T::B_N>{}, 1_I), p_coord_c);
            auto part = [&](int s_) { return make_gmem(ws_tiles + (static_cast<long>(tile_id) * split_k + s_) * tile_elems,
                                                       static_cast<unsigned int>(tile_elems * 4)); };
            auto g_own = part(k_split);
            const auto wc_offs = opus::layout_to_offsets<T::VEC_C>(u_wc);   // per fragment (16 B per lane)
            static_assert(MMA::mma_c_len == T::VEC_C);
            opus::static_for<2>([&](auto hm) { opus::static_for<2>([&](auto hn) { opus::static_for<T::E_M * T::E_N>([&](auto f) {
                store<T::VEC_C>(g_own, v_c[hm.value][hn.value][f.value], wc_offs[f.value] + (hm.value * T::HALF_B_M * T::B_N + hn.value * T::HALF_B_N));
            }); }); });
            __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");    // partial visible device-wide before the counter
            s_waitcnt_vmcnt(0_I);
            __builtin_amdgcn_s_barrier();
            if (thread_id_x() == 0) {
                const int arrived = __scoped_atomic_fetch_add(&counters[tile_id], 1, __ATOMIC_ACQ_REL, __MEMORY_SCOPE_DEVICE);
                const int last = arrived == split_k - 1;
                if (last) __scoped_atomic_store_n(&counters[tile_id], 0, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);   // ready for the next launch
                *sk_last_p = last;
            }
            s_waitcnt_lgkmcnt(0_I);
            __builtin_amdgcn_s_barrier();
            if (!*sk_last_p) return;
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");    // see the other blocks' partials
            for (int s2 = 0; s2 < split_k; ++s2) {
                if (s2 == k_split) continue;
                auto g_p = part(s2);
                opus::static_for<2>([&](auto hm) { opus::static_for<2>([&](auto hn) { opus::static_for<T::E_M * T::E_N>([&](auto f) {
                    const auto q = load<T::VEC_C>(g_p, wc_offs[f.value] + (hm.value * T::HALF_B_M * T::B_N + hn.value * T::HALF_B_N));
                    opus::static_for<MMA::mma_c_len>([&](auto j) { v_c[hm.value][hn.value][f.value][j.value] += q[j.value]; });
                }); }); });
            }
        }
    }
    stage_quadrant(v_c[0][0], 0);
    stage_quadrant(v_c[0][1], 1);
    sync_lds();
    copy_output_half(0);
    sync_lds();
    stage_quadrant(v_c[1][0], 0);
    stage_quadrant(v_c[1][1], 1);
    sync_lds();
    if constexpr (!T::PERSIST) {   // one tile per block: bottom half straight out
        copy_output_half(1);
    } else {
    opus::static_for<c_copies>([&](auto i) {   // h1: bottom half -> registers, stored in the next prologue
        if constexpr (T::B_N == 192) {
            v_pend_c[i.value] = load<VEC_OUT>(s_c, c_row(output_thread_id, i) * c_lds_stride + c_col(output_thread_id, i));
        } else {
        const int linear = output_thread_id * VEC_OUT + decltype(i)::value * T::BLOCK_SIZE * VEC_OUT;
        v_pend_c[i.value] = c_unswz(load<VEC_OUT>(s_c, (linear / T::B_N) * c_lds_stride + linear % T::B_N), i);
        }
    });
    pend_c_ptr = reinterpret_cast<D_C*>(kargs.ptr_c) + batch_id * kargs.stride_c_batch + row * kargs.stride_c + col;
    pend_c = 1;
    pend_rows = rows_in;
    pend_cols = cols_in;
    }
    // Next tile's prologue copies overwrite the A+B LDS used for staging C: every wave must
    // have finished its staged-C reads.
    s_waitcnt_lgkmcnt(0_I);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);
    }   // persistent tile loop
    if (T::PERSIST && pend_c) {   // h1: the last tile's bottom half
        unsigned int lm = ~0u; asm volatile("" : "+s"(lm));
        const int ln = static_cast<int>(__builtin_amdgcn_mbcnt_hi(lm, __builtin_amdgcn_mbcnt_lo(lm, 0u)));
        s_waitcnt_lgkmcnt(0_I);
        store_pend_c(wave_id * T::WARP_SIZE + ln);
    }
#endif   // __gfx950__
}

template<class Traits>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, 1)
void gemm_a4w4_mxfp4_nt_kernel(opus_gemm_mxfp4_scale_kargs_gfx950 kargs) {
    gemm_a4w4_mxfp4_nt_body<Traits>(kargs);
}
