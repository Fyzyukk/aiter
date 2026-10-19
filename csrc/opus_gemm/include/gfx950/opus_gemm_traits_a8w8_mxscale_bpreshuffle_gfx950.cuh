// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <climits>
#include <type_traits>
#include "opus_gemm_utils.cuh"
#include "opus_gemm_traits_a8w8_scale_gfx950.cuh"

struct opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 {
    const void *__restrict__ ptr_a;
    const void *__restrict__ ptr_b;
    void *__restrict__ ptr_c;
    int m;
    int n;
    int k;
    int batch;
    int stride_a;
    int stride_b;
    int stride_c;
    int stride_a_batch;
    int stride_b_batch;
    int stride_c_batch;
    const void *__restrict__ ptr_sfa;
    const void *__restrict__ ptr_sfb;
    int stride_sfa;
    int stride_sfb;
    int stride_sfa_batch;
    int stride_sfb_batch;
};
using opus_gemm_mxscale_bpreshuffle_kargs_gfx950 =
    opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950;
static_assert(sizeof(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950) == 96);

#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__) && defined(OPUS_BP_FIXED_M) &&          \
    defined(OPUS_BP_FIXED_N) && defined(OPUS_BP_FIXED_K)
template <class Arguments>
__device__ __forceinline__ void
opus_gemm_blockscale_mxscale_bpreshuffle_apply_fixed_args(Arguments &a) {
    if constexpr (requires { a.m; })
        a.m = OPUS_BP_FIXED_M;
    if constexpr (requires { a.n; })
        a.n = OPUS_BP_FIXED_N;
    if constexpr (requires { a.k; })
        a.k = OPUS_BP_FIXED_K;
    if constexpr (requires { a.batch; })
        a.batch = 1;
    if constexpr (requires { a.stride_a; })
        a.stride_a = OPUS_BP_FIXED_K;
    if constexpr (requires { a.stride_b; })
        a.stride_b = OPUS_BP_FIXED_K;
    if constexpr (requires { a.stride_c; })
        a.stride_c = OPUS_BP_FIXED_N;
    if constexpr (requires { a.stride_a_batch; })
        a.stride_a_batch = static_cast<int>(int64_t(OPUS_BP_FIXED_M) * OPUS_BP_FIXED_K);
    if constexpr (requires { a.stride_b_batch; })
        a.stride_b_batch = static_cast<int>(int64_t(OPUS_BP_FIXED_N) * OPUS_BP_FIXED_K);
    if constexpr (requires { a.stride_c_batch; })
        a.stride_c_batch = static_cast<int>(int64_t(OPUS_BP_FIXED_M) * OPUS_BP_FIXED_N);
    if constexpr (requires { a.stride_sfa; })
        a.stride_sfa = OPUS_BP_FIXED_M;
    if constexpr (requires { a.stride_sfb; })
        a.stride_sfb = OPUS_BP_FIXED_K / 128;
    if constexpr (requires { a.stride_sfa_batch; })
        a.stride_sfa_batch = OPUS_BP_FIXED_M * (OPUS_BP_FIXED_K / 128);
    if constexpr (requires { a.stride_sfb_batch; })
        a.stride_sfb_batch = (OPUS_BP_FIXED_N / 128) * (OPUS_BP_FIXED_K / 128);
}
#endif

#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
namespace opus_gemm_blockscale_mxscale_bpreshuffle_panel_half_panel {
template <class T, class Gmem>
__device__ inline auto load_sfa_vector(Gmem &g_sfa, int gmem_offset, int valid_rows) {
    opus::vector_t<unsigned char, T::VEC_SCALE_A> raw;
    if (valid_rows >= T::VEC_SCALE_A && gmem_offset % T::VEC_SCALE_A == 0) {
        raw = opus::load<T::VEC_SCALE_A>(g_sfa, gmem_offset);
    } else {
        opus::static_for<T::VEC_SCALE_A>([&](auto byte_i) {
            constexpr int byte = decltype(byte_i)::value;
            raw[byte] = 0x7f;
            if (byte < valid_rows)
                raw[byte] = opus::load<1>(g_sfa, gmem_offset + byte)[0];
        });
    }
    return raw;
}
}

// Shared output packing used by the retained B-direct and LDS-panel pipelines.
namespace opus_gemm_blockscale_mxscale_bpreshuffle_register_prefetch_mfma16_output {
template <class T, bool ContiguousWaveN = false, class Accumulators, class Gmem>
__device__ __forceinline__ void store_mfma16_packed(const Accumulators &c, Gmem gc, int lane,
                                                    int wm, int wn, int stride_c) {
    using namespace opus;
    static_assert(T::W_M == 16 && T::W_N == 16);
    const int side = (lane / 16) % 2;
    static_for<T::E_M * T::E_N / 2>([&](auto pair) {
        constexpr int ci = decltype(pair)::value * 2;
        const auto first = __builtin_bit_cast(vector_t<unsigned, 2>, cast<bf16_t>(c[ci]));
        const auto second = __builtin_bit_cast(vector_t<unsigned, 2>, cast<bf16_t>(c[ci + 1]));
        const unsigned own0 = side ? second[0] : first[0];
        const unsigned own1 = side ? second[1] : first[1];
        const unsigned src0 = side ? first[0] : second[0];
        const unsigned src1 = side ? first[1] : second[1];
        const unsigned other0 = __builtin_amdgcn_ds_bpermute((lane ^ 16) * 4, src0);
        const unsigned other1 = __builtin_amdgcn_ds_bpermute((lane ^ 16) * 4, src1);
        vector_t<unsigned, 4> packed;
        packed[0] = side ? other0 : own0;
        packed[1] = side ? other1 : own1;
        packed[2] = side ? own0 : other0;
        packed[3] = side ? own1 : other1;
        const int m = (ci + side) / T::E_N, n = (ci + side) % T::E_N;
        const int rr = (m * T::T_M + wm) * 16 + lane % 16;
        const int nn = (ContiguousWaveN ? n + wn * T::E_N : n * T::T_N + wn) * 16 + (lane / 32) * 8;
        store<8>(gc, __builtin_bit_cast(vector_t<bf16_t, 8>, packed), rr * stride_c + nn, 0,
                 number<T::STORE_CACHE>{});
    });
    if constexpr (T::E_M * T::E_N % 2 != 0) {
        constexpr int ci = T::E_M * T::E_N - 1;
        const auto words = __builtin_bit_cast(vector_t<unsigned, 2>, cast<bf16_t>(c[ci]));
        const unsigned other0 = __builtin_amdgcn_ds_bpermute((lane ^ 16) * 4, words[0]);
        const unsigned other1 = __builtin_amdgcn_ds_bpermute((lane ^ 16) * 4, words[1]);
        if (side == 0) {
            vector_t<unsigned, 4> packed{words[0], words[1], other0, other1};
            const int rr = ((ci / T::E_N) * T::T_M + wm) * 16 + lane % 16;
            const int nn = (ContiguousWaveN ? (ci % T::E_N) + wn * T::E_N : (ci % T::E_N) * T::T_N + wn) * 16 + (lane / 32) * 8;
            store<8>(gc, __builtin_bit_cast(vector_t<bf16_t, 8>, packed), rr * stride_c + nn, 0,
                     number<T::STORE_CACHE>{});
        }
    }
}
} // namespace opus_gemm_blockscale_mxscale_bpreshuffle_register_prefetch_mfma16_output

template <class T>
__device__ inline auto
opus_gemm_blockscale_mxscale_bpreshuffle_tiled_make_layout_ga_scale(int lane_id, int wave_id_m,
                                                                    int wave_id_n, int stride_a) {
    constexpr int threads_k = T::B_K / T::VEC_A;
    constexpr int threads_m_per_block = T::BLOCK_SIZE / threads_k;
    constexpr int threads_m_per_wave = T::WARP_SIZE / threads_k;
    constexpr auto ga_block_shape =
        opus::make_tuple(opus::number<T::HALF_B_M / threads_m_per_block>{}, opus::number<T::T_N>{},
                         opus::number<threads_m_per_wave>{}, opus::number<T::T_M>{},
                         opus::number<threads_k>{}, opus::number<T::VEC_A>{});
    constexpr auto ga_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));
    return opus::make_layout<T::VEC_A>(
        ga_block_shape,
        opus::unfold_x_stride(ga_block_dim, ga_block_shape, opus::tuple{stride_a, 1_I}),
        opus::unfold_p_coord(ga_block_dim, opus::tuple{wave_id_n, lane_id / threads_k, wave_id_m,
                                                       lane_id % threads_k}));
}

template <class T>
__device__ inline constexpr auto
opus_gemm_blockscale_mxscale_bpreshuffle_tiled_make_layout_gb_scale(int lane_id, int wave_id_m,
                                                                    int wave_id_n, int stride_b) {
    constexpr int lanes_n = T::W_N;
    constexpr int lanes_k = T::WARP_SIZE / lanes_n;
    constexpr int k_per_wave_load = lanes_k * T::VEC_B;
    constexpr int n_repeats = T::HALF_B_N / (T::NUM_WAVES * lanes_n);
    constexpr int k_repeats = T::B_K / k_per_wave_load;
    constexpr auto gb_block_shape = opus::make_tuple(
        opus::number<T::T_N>{}, opus::number<T::T_M>{}, opus::number<n_repeats>{},
        opus::number<k_repeats>{}, opus::number<T::WARP_SIZE>{}, opus::number<T::VEC_B>{});
    constexpr auto gb_block_dim =
        opus::make_tuple(opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}),
                         opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));
    return opus::make_layout<T::VEC_B>(
        gb_block_shape,
        opus::unfold_x_stride(gb_block_dim, gb_block_shape, opus::tuple{T::W_N * stride_b, 1_I}),
        opus::unfold_p_coord(gb_block_dim, opus::tuple{wave_id_n, wave_id_m, lane_id}));
}

template <class T>
__device__ inline constexpr auto
opus_gemm_blockscale_mxscale_bpreshuffle_tiled_make_layout_ra_scale(int lane_id, int wave_id_m) {
    constexpr auto ra_block_shape =
        opus::make_tuple(opus::number<T::E_M>{}, opus::number<T::T_M>{}, opus::number<T::T_M>{},
                         opus::number<T::W_M / T::T_M>{}, opus::number<T::E_K>{},
                         opus::number<T::W_M * T::W_K / T::WARP_SIZE / T::VEC_A>{},
                         opus::number<T::WARP_SIZE / T::W_M>{}, opus::number<T::VEC_A>{});
    constexpr auto ra_block_dim =
        opus::make_tuple(opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
                         opus::make_tuple(opus::p_dim{}, opus::y_dim{}, opus::y_dim{},
                                          opus::p_dim{}, opus::y_dim{}));
    const int lane_id_m = lane_id % T::W_M;
    return opus::make_layout<T::VEC_A>(
        ra_block_shape,
        opus::unfold_x_stride(
            ra_block_dim, ra_block_shape,
            opus::tuple{opus::number<T::smem_linear_wave + T::smem_padding>{}, 1_I}),
        opus::unfold_p_coord(ra_block_dim, opus::tuple{wave_id_m, lane_id_m % T::T_M,
                                                       lane_id_m / T::T_M, lane_id / T::W_M}));
}

template <class T>
__device__ inline constexpr auto
opus_gemm_blockscale_mxscale_bpreshuffle_tiled_make_layout_rb_scale(int lane_id, int wave_id_n) {
    constexpr int k_vectors = T::W_N * T::W_K / (T::WARP_SIZE * T::VEC_B);
    constexpr auto rb_block_shape = opus::make_tuple(
        opus::number<T::E_N>{}, opus::number<T::T_N>{}, opus::number<T::E_K>{},
        opus::number<k_vectors>{}, opus::number<T::WARP_SIZE>{}, opus::number<T::VEC_B>{});
    constexpr auto rb_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}, opus::y_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));
    return opus::make_layout<T::VEC_B>(
        rb_block_shape,
        opus::unfold_x_stride(
            rb_block_dim, rb_block_shape,
            opus::tuple{opus::number<T::smem_linear_wave + T::smem_padding>{}, 1_I}),
        opus::unfold_p_coord(rb_block_dim, opus::tuple{wave_id_n, lane_id}));
}

template <class T>
__device__ inline auto
opus_gemm_blockscale_mxscale_bpreshuffle_tiled_make_layout_sa_scale(int wave_id_m, int wave_id_n) {
    constexpr int num_waves = T::BLOCK_SIZE / T::WARP_SIZE;
    constexpr auto sa_block_shape =
        opus::make_tuple(opus::number<T::smem_m_rep / num_waves>{}, opus::number<T::T_N>{},
                         opus::number<T::T_M>{}, opus::number<T::VEC_A>{});
    constexpr auto sa_block_dim =
        opus::make_tuple(opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
                         opus::make_tuple(opus::y_dim{}));
    return opus::make_layout(
        sa_block_shape,
        opus::unfold_x_stride(
            sa_block_dim, sa_block_shape,
            opus::tuple{opus::number<T::smem_linear_wave + T::smem_padding>{}, 1_I}),
        opus::unfold_p_coord(sa_block_dim, opus::tuple{wave_id_n, wave_id_m}));
}

template <class T>
__device__ inline constexpr auto
opus_gemm_blockscale_mxscale_bpreshuffle_tiled_make_layout_sb_scale(int wave_id_m, int wave_id_n) {
    constexpr int n_repeats = T::HALF_B_N / (T::NUM_WAVES * T::W_N);
    constexpr int k_per_wave_load = (T::WARP_SIZE / T::W_N) * T::VEC_B;
    constexpr int k_repeats = T::B_K / k_per_wave_load;
    constexpr auto sb_block_shape =
        opus::make_tuple(opus::number<T::T_N>{}, opus::number<T::T_M>{}, opus::number<n_repeats>{},
                         opus::number<k_repeats>{}, opus::number<T::VEC_B>{});
    constexpr auto sb_block_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}, opus::y_dim{}),
        opus::make_tuple(opus::y_dim{}));
    return opus::make_layout(
        sb_block_shape,
        opus::unfold_x_stride(
            sb_block_dim, sb_block_shape,
            opus::tuple{opus::number<T::smem_linear_wave + T::smem_padding>{}, 1_I}),
        opus::unfold_p_coord(sb_block_dim, opus::tuple{wave_id_n, wave_id_m}));
}
#endif

constexpr int opus_gemm_blockscale_mxscale_bpreshuffle_output_rows(int bm, int bn,
                                                                   int pipeline_bytes) {
    for (int rows = bm; rows >= 16; rows -= 16)
        if (bm % rows == 0 && rows * (bn + 8) * 2 <= pipeline_bytes)
            return rows;
    return 16;
}
template <int BM, int BN, int NB, int BD, int BC = 0, bool XCD = false, int FixedK = 7168,
          int WN = 4, int WGPerCU = 1, int ACache = 0, int StoreCache = 2, int Output = 0, int BlockK = 256,
          bool Contiguous = false, bool PinA = false, bool PinC = false, int GroupM = 0>
struct opus_gemm_blockscale_mxscale_bpreshuffle_b_direct_traits {
    static constexpr int WG_PER_CU = WGPerCU, CACHECTL_A = ACache, STORE_CACHE = StoreCache;
    static constexpr int B_M = BM, B_N = BN, B_K = BlockK;
    static constexpr int T_M = 1, T_N = WN, T_K = 1, BLOCK_SIZE = WN * 64;
    static constexpr int K = FixedK, PANELS = K / B_K, SCALE_COLS = K / 128;
    static constexpr int MR = BM / 16, NR = BN / (WN * 16);
    static constexpr int OUTPUT = Output, E_M = MR, E_N = NR, W_M = 16, W_N = 16;
    static constexpr bool CONTIGUOUS = Contiguous, PIN_A = PinA, PIN_C = PinC;
    static constexpr int GROUPED_M = GroupM;
    static_assert(GroupM == 0 || GroupM == 4 || GroupM == 8);
    static_assert(!PinA || (B_K / 128) * MR * 8 <= 128);
    static_assert(!PinC || MR * NR * 4 <= 128);
    static_assert(Output == 0 || Output == 2);
    static constexpr int STAGES = NB, B_AHEAD = BD, B_SETS = BD + 1, B_CACHE = BC;
    static constexpr bool XCD_ORDER = XCD, SB_SLOTS = SCALE_COLS % 4 != 0, C_AGPR = false, PERSTAGE = false;

    // The A ring and resident byte-scale panels share LDS with the output tile.
    static constexpr int A_BYTES = BM * B_K, STAGE_BYTES = A_BYTES;
    static constexpr int A_COPIES = A_BYTES / (BLOCK_SIZE * 16);
    static constexpr int SA_ROWS = (BM + 15) / 16 * 16;
    static constexpr int SA_COPIES =
        (SA_ROWS * SCALE_COLS + BLOCK_SIZE * 16 - 1) / (BLOCK_SIZE * 16);
    static constexpr int SA_BYTES = SA_COPIES * BLOCK_SIZE * 16;
    static constexpr int SB_ROWS = (BN + 127) / 128;
    static constexpr int SB_BYTES = BLOCK_SIZE * 4;
    static constexpr int OFF_SA = NB * STAGE_BYTES, OFF_SB = OFF_SA + SA_BYTES;
    static constexpr int PIPELINE_BYTES = OFF_SB + SB_BYTES;
    static constexpr int C_ROWS =
        opus_gemm_blockscale_mxscale_bpreshuffle_output_rows(BM, BN, PIPELINE_BYTES);
    static constexpr int LDS_BYTES = PIPELINE_BYTES;

    static constexpr int INITIAL_A = NB - 1, INITIAL_B = BD;
    static constexpr int STEADY = PANELS - INITIAL_A;
    static constexpr int B_ISSUES = NR * 2 * (B_K / 128);
    static constexpr int SCALE_ISSUES = SA_COPIES + 1;

    // Counts follow the actual prologue: B, first A, scales, remaining A.
    static constexpr int wait_a(int t) {
        int issued = INITIAL_B * B_ISSUES;
        int ready = -1;
        issued += A_COPIES;
        if (t == 0)
            ready = issued;
        issued += SCALE_ISSUES;
        // K0 consumes the resident scales as well as A0. The first barrier
        // must publish those async LDS writes before any scale LDS reads.
        if (t == 0)
            ready = issued;
        for (int stage = 1; stage < INITIAL_A; ++stage) {
            issued += A_COPIES;
            if (stage == t)
                ready = issued;
        }
        for (int step = 0; step < t; ++step) {
            if (step + BD < PANELS)
                issued += B_ISSUES;
            if (step + NB - 1 < PANELS) {
                issued += A_COPIES;
                if (step + NB - 1 == t)
                    ready = issued;
            }
        }
        return issued - ready;
    }
    static constexpr int wait_b(int t) {
        int issued = 0, ready = -1;
        for (int set = 0; set < INITIAL_B; ++set) {
            issued += B_ISSUES;
            if (set == t)
                ready = issued;
        }
        issued += INITIAL_A * A_COPIES + SCALE_ISSUES;
        for (int step = 0; step <= t; ++step) {
            if (step + BD < PANELS) {
                issued += B_ISSUES;
                if (step + BD == t)
                    ready = issued;
            }
            if (step + NB - 1 < PANELS)
                issued += A_COPIES;
        }
        return issued - ready;
    }

    static_assert(BM > 0 && BM % 16 == 0 && BN > 0 && BN % (WN * 16) == 0);
    static_assert(WN == 1 || WN == 2 || WN == 4 || WN == 8);
    static_assert(NB >= 2 && BD >= 1 && BD < NB);
    static_assert((BC == 0 || BC == 1 || BC == 17) && K > 0 && K % B_K == 0);
    static_assert(ACache == 0 || ACache == 1 || ACache == 17);
    static_assert(BN <= 256 && (WGPerCU == 1 || WGPerCU == 2));
    static_assert(B_K == 128 || B_K == 256);
    static_assert(A_BYTES % (BLOCK_SIZE * 16) == 0 && PANELS >= INITIAL_A);
    static_assert(SB_ROWS * (SB_SLOTS ? SCALE_COLS : SCALE_COLS / 4) <= BLOCK_SIZE);
    static_assert(((LDS_BYTES + 1279) / 1280 * 1280) * WGPerCU <= 160 * 1024);
    static_assert(MR * NR * 4 <= 256);
};
struct opus_gemm_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_common {
    static constexpr int VEC_SF = 4;
    static constexpr int VEC_SF_PAIR = 2 * VEC_SF;
};

template <bool PadM = false>
struct opus_gemm_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_4wave_pin_agpr_traits
    : opus_gemm_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_common {
    static constexpr bool PAD_M = PadM;
    static constexpr int BLOCK_SIZE = 256;
    static constexpr int WARP_SIZE = 64;
    static constexpr int NUM_WAVES = BLOCK_SIZE / WARP_SIZE;
    static constexpr int MIN_WGS_PER_CU = 1;
    static constexpr int B_M = 256;
    static constexpr int B_N = 256;
    static constexpr int B_K = 128;
    static constexpr int T_M = 2;
    static constexpr int T_N = 2;
    static constexpr int T_K = 1;
    static constexpr int W_M = 16;
    static constexpr int W_N = 16;
    static constexpr int W_K = 128;
    static constexpr int HALF_B_M = B_M / 2;
    static constexpr int HALF_B_N = B_N / 2;
    static_assert(NUM_WAVES == 4);
    static_assert(NUM_WAVES == T_M * T_N * T_K);
    static_assert(HALF_B_M % (W_M * T_M) == 0);
    static_assert(HALF_B_N % (W_N * T_N) == 0);
    static_assert(B_K % (W_K * T_K) == 0);
    static constexpr int E_M = HALF_B_M / (W_M * T_M);
    static constexpr int E_N = HALF_B_N / (W_N * T_N);
    static constexpr int E_K = B_K / (W_K * T_K);
    static constexpr int VEC_A = 16;
    static constexpr int VEC_B = 16;
    static constexpr int VEC_C = 4;
    static constexpr int OUTPUT_TILES_PER_WG = 1;
    static constexpr int VEC_SCALE_A = 16;
    static constexpr int VEC_SCALE_SF = 4;
    static constexpr int VEC_SCALE_SF_PAIR = 8;
    static constexpr int GROUP_M = 1;
    static constexpr int GROUP_N = 128;
    static constexpr int GROUP_K = 128;
    static constexpr int MFMA_SCALE_GROUP_K = 32;
    static constexpr int SCALE_KGROUPS_PER_MFMA = W_K / MFMA_SCALE_GROUP_K;
    static constexpr int NUM_KGROUPS = B_K / MFMA_SCALE_GROUP_K;
    static constexpr int smem_linear_wave = WARP_SIZE * VEC_A;
    static constexpr int smem_sub = smem_linear_wave / B_K;
    static constexpr int smem_m_rep = HALF_B_M / smem_sub;
    static constexpr int smem_n_rep = HALF_B_N / smem_sub;
    static constexpr int smem_padding = 32;
    static constexpr int SCALE_N_HALVES = B_N / HALF_B_N;
    static_assert(NUM_KGROUPS == 4);
    static constexpr int a_buffer_load_insts = HALF_B_M * B_K / (BLOCK_SIZE * VEC_A);
    static constexpr int b_buffer_load_insts = HALF_B_N * B_K / (BLOCK_SIZE * VEC_B);
    static constexpr int a_ds_read_insts = E_M * W_M * W_K / (WARP_SIZE * VEC_A);
    static constexpr int b_ds_read_insts = E_N * W_N * W_K / (WARP_SIZE * VEC_B);
    static constexpr int SMEM_A_ELEMS = smem_m_rep * (smem_linear_wave + smem_padding) * 4;
    static constexpr int SMEM_B_ELEMS = smem_n_rep * (smem_linear_wave + smem_padding) * 4;
    static constexpr int SCALE_PANEL_K_CAPACITY = 64;
    static constexpr int SFA_K_COLUMNS_PER_WAVE = WARP_SIZE / W_M;
    static constexpr int SFA_K_COLUMNS_PER_PASS = NUM_WAVES * SFA_K_COLUMNS_PER_WAVE;
    static constexpr int SFA_PASSES_PER_CACHE_PANEL =
        SCALE_PANEL_K_CAPACITY / SFA_K_COLUMNS_PER_PASS;
    static constexpr int SFA_PANEL_PITCH = B_M;
    static constexpr int SFA_PANEL_BYTES = SFA_PANEL_PITCH * SCALE_PANEL_K_CAPACITY;
    static constexpr int SFB_PANEL_BYTES = SCALE_N_HALVES * SCALE_PANEL_K_CAPACITY * VEC_SF;
    static constexpr int LDS_BYTES =
        SMEM_A_ELEMS + SMEM_B_ELEMS + SFA_PANEL_BYTES + SFB_PANEL_BYTES;
    static_assert(E_M == 4 && E_N == 4 && E_K == 1);
    static_assert(VEC_SCALE_SF == E_M);
    static_assert(VEC_SCALE_SF_PAIR == 2 * VEC_SCALE_SF);
    static_assert(SFA_PASSES_PER_CACHE_PANEL * SFA_K_COLUMNS_PER_PASS == SCALE_PANEL_K_CAPACITY);
    static_assert(a_ds_read_insts == 8 && b_ds_read_insts == 8);
    static_assert(SFA_PANEL_BYTES == 16384 && SFB_PANEL_BYTES == 512);
    static_assert(LDS_BYTES == 152064 && LDS_BYTES <= 163840);
};

template <class Base, int Panel>
struct opus_gemm_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_scale_panel_traits : Base {
    static_assert(Panel == 16 || Panel == 32 || Panel == 64);
    static constexpr int SCALE_PANEL_K_CAPACITY = Panel;
    static constexpr int SFA_PASSES_PER_CACHE_PANEL = Panel / Base::SFA_K_COLUMNS_PER_PASS;
    static constexpr int SFA_PANEL_BYTES = Base::SFA_PANEL_PITCH * Panel;
    static constexpr int SFB_PANEL_BYTES = Base::SCALE_N_HALVES * Panel * Base::VEC_SF;
    static constexpr int LDS_BYTES =
        Base::SMEM_A_ELEMS + Base::SMEM_B_ELEMS + SFA_PANEL_BYTES + SFB_PANEL_BYTES;
};

template <auto OpusValue0>
constexpr int opus_gemm_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_A0_AGPR_BASE = OpusValue0;

namespace opus_gemm_blockscale_mxscale_bpreshuffle_pin_detail {

#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
template <int Vec, int CHUNK_COUNT, int AGPR_BASE, class Chunk, class Mem, class Layout>
__device__ inline void load_operand_chunks_pinned(Chunk *chunks, Mem &smem, const Layout &layout) {
    constexpr int dwords_per_chunk = sizeof(Chunk) / sizeof(opus::u32_t);
    static_assert(AGPR_BASE >= 0);
    static_assert(AGPR_BASE + CHUNK_COUNT * dwords_per_chunk <= 128);
    const auto offsets = opus::layout_to_offsets<Vec>(layout);
    opus::static_for<CHUNK_COUNT>([&](auto i) {
        constexpr int chunk = decltype(i)::value;
        [[clang::amdgpu_pin_agpr(AGPR_BASE + chunk * dwords_per_chunk)]] chunks[chunk] =
            opus::load<Vec>(smem, offsets[chunk]);
    });
}

template <int Vec, int OFFSET_BEGIN, class Reg, class Fragment, class Mem, class Offsets>
__device__ inline void load_operand_fragment_pinned(Reg reg, Fragment &fragment, Mem &smem,
                                                    const Offsets &offsets) {
    using Chunk = decltype(opus::load<Vec>(smem, offsets[OFFSET_BEGIN]));
    constexpr int dwords_per_chunk = sizeof(Chunk) / sizeof(opus::u32_t);
    constexpr int physical_reg = decltype(reg)::value;
    static_assert(physical_reg >= 0 && physical_reg < 128);
    opus::array<Chunk, 2> values;
    opus::static_for<2>([&](auto i) {
        constexpr int chunk = decltype(i)::value;
        [[clang::amdgpu_pin_agpr(physical_reg + chunk * dwords_per_chunk)]] values[chunk] =
            opus::load<Vec>(smem, offsets[OFFSET_BEGIN + chunk]);
    });
    Fragment result;
    opus::set_slice(result, values[0], opus::number<0>{}, opus::number<16>{});
    opus::set_slice(result, values[1], opus::number<16>{}, opus::number<32>{});
    [[clang::amdgpu_pin_agpr(physical_reg)]] fragment = result;
}

template <int Vec, int OFFSET_BEGIN, class Fragment, class Mem, class Offsets>
__device__ inline Fragment load_operand_fragment_staged(Mem &smem, const Offsets &offsets) {
    using Chunk = decltype(opus::load<Vec>(smem, offsets[OFFSET_BEGIN]));
    opus::array<Chunk, 2> values;
    opus::static_for<2>([&](auto i) {
        constexpr int chunk = decltype(i)::value;
        values[chunk] = opus::load<Vec>(smem, offsets[OFFSET_BEGIN + chunk]);
    });
    Fragment result;
    opus::set_slice(result, values[0], opus::number<0>{}, opus::number<16>{});
    opus::set_slice(result, values[1], opus::number<16>{}, opus::number<32>{});
    return result;
}

template <class T>
__device__ inline auto make_layout_ga_scale(int lane_id, int wave_id_m, int wave_id_n,
                                            int stride_a) {
    constexpr int threads_k = T::B_K / T::VEC_A;
    constexpr int threads_m_per_block = T::BLOCK_SIZE / threads_k;
    constexpr int threads_m_per_wave = T::WARP_SIZE / threads_k;
    constexpr auto ga_block_shape =
        opus::make_tuple(opus::number<T::HALF_B_M / threads_m_per_block>{}, opus::number<T::T_N>{},
                         opus::number<threads_m_per_wave>{}, opus::number<T::T_M>{},
                         opus::number<threads_k>{}, opus::number<T::VEC_A>{});
    constexpr auto ga_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));
    return opus::make_layout<T::VEC_A>(
        ga_block_shape,
        opus::unfold_x_stride(ga_block_dim, ga_block_shape, opus::tuple{stride_a, 1_I}),
        opus::unfold_p_coord(ga_block_dim, opus::tuple{wave_id_n, lane_id / threads_k, wave_id_m,
                                                       lane_id % threads_k}));
}

template <class T>
__device__ inline constexpr auto make_layout_gb_scale(int lane_id, int wave_id_m, int wave_id_n,
                                                      int stride_b) {
    constexpr int lanes_n = T::W_N;
    constexpr int lanes_k = T::WARP_SIZE / lanes_n;
    constexpr int k_per_wave_load = lanes_k * T::VEC_B;
    constexpr int n_repeats = T::HALF_B_N / (T::NUM_WAVES * lanes_n);
    constexpr int k_repeats = T::B_K / k_per_wave_load;
    constexpr auto gb_block_shape = opus::make_tuple(
        opus::number<T::T_N>{}, opus::number<T::T_M>{}, opus::number<n_repeats>{},
        opus::number<k_repeats>{}, opus::number<T::WARP_SIZE>{}, opus::number<T::VEC_B>{});
    constexpr auto gb_block_dim =
        opus::make_tuple(opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}),
                         opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));
    return opus::make_layout<T::VEC_B>(
        gb_block_shape,
        opus::unfold_x_stride(gb_block_dim, gb_block_shape, opus::tuple{T::W_N * stride_b, 1_I}),
        opus::unfold_p_coord(gb_block_dim, opus::tuple{wave_id_n, wave_id_m, lane_id}));
}

template <class T>
__device__ inline constexpr auto make_layout_gsfa_scale(int lane_id, int wave_id_m, int wave_id_n,
                                                        int stride_sfa) {
    constexpr auto block_shape =
        opus::make_tuple(opus::number<T::SFA_PASSES_PER_CACHE_PANEL>{}, opus::number<T::T_N>{},
                         opus::number<T::T_M>{}, opus::number<T::SFA_K_COLUMNS_PER_WAVE>{},
                         opus::number<T::B_M / T::VEC_SCALE_A>{}, opus::number<T::VEC_SCALE_A>{});
    constexpr auto block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));
    const int lane_id_m = (lane_id & 8) | ((lane_id & 3) << 1) | ((lane_id & 4) >> 2);
    return opus::make_layout<T::VEC_SCALE_A>(
        block_shape, opus::unfold_x_stride(block_dim, block_shape, opus::tuple{stride_sfa, 1_I}),
        opus::unfold_p_coord(block_dim,
                             opus::tuple{wave_id_n, wave_id_m, lane_id / T::W_M, lane_id_m}));
}

template <class T>
__device__ inline constexpr auto make_layout_gsfb_scale(int lane_id, int wave_id) {
    constexpr auto block_shape =
        opus::make_tuple(opus::number<T::SCALE_N_HALVES>{},
                         opus::number<T::SCALE_PANEL_K_CAPACITY>{}, opus::number<1>{});
    constexpr auto block_dim = opus::make_tuple(opus::make_tuple(opus::p_dim{}),
                                                opus::make_tuple(opus::p_dim{}, opus::y_dim{}));
    return opus::make_layout<1>(
        block_shape, opus::unfold_x_stride(block_dim, block_shape, opus::tuple{0_I, 1_I}),
        opus::unfold_p_coord(block_dim, opus::tuple{wave_id, lane_id}));
}

template <class T>
__device__ inline constexpr auto make_layout_ra_scale(int lane_id, int wave_id_m) {
    constexpr auto ra_block_shape =
        opus::make_tuple(opus::number<T::E_M>{}, opus::number<T::T_M>{}, opus::number<T::T_M>{},
                         opus::number<T::W_M / T::T_M>{}, opus::number<T::E_K>{},
                         opus::number<T::W_M * T::W_K / T::WARP_SIZE / T::VEC_A>{},
                         opus::number<T::WARP_SIZE / T::W_M>{}, opus::number<T::VEC_A>{});
    constexpr auto ra_block_dim =
        opus::make_tuple(opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
                         opus::make_tuple(opus::p_dim{}, opus::y_dim{}, opus::y_dim{},
                                          opus::p_dim{}, opus::y_dim{}));
    const int lane_id_m = lane_id % T::W_M;
    return opus::make_layout<T::VEC_A>(
        ra_block_shape,
        opus::unfold_x_stride(
            ra_block_dim, ra_block_shape,
            opus::tuple{opus::number<T::smem_linear_wave + T::smem_padding>{}, 1_I}),
        opus::unfold_p_coord(ra_block_dim, opus::tuple{wave_id_m, lane_id_m % T::T_M,
                                                       lane_id_m / T::T_M, lane_id / T::W_M}));
}

template <class T>
__device__ inline constexpr auto make_layout_rb_scale(int lane_id, int wave_id_n) {
    constexpr int k_vectors = T::W_N * T::W_K / (T::WARP_SIZE * T::VEC_B);
    constexpr auto rb_block_shape = opus::make_tuple(
        opus::number<T::E_N>{}, opus::number<T::T_N>{}, opus::number<T::E_K>{},
        opus::number<k_vectors>{}, opus::number<T::WARP_SIZE>{}, opus::number<T::VEC_B>{});
    constexpr auto rb_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}, opus::y_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));
    return opus::make_layout<T::VEC_B>(
        rb_block_shape,
        opus::unfold_x_stride(
            rb_block_dim, rb_block_shape,
            opus::tuple{opus::number<T::smem_linear_wave + T::smem_padding>{}, 1_I}),
        opus::unfold_p_coord(rb_block_dim, opus::tuple{wave_id_n, lane_id}));
}

template <class T, int Vec = T::VEC_SCALE_SF>
__device__ inline constexpr auto make_layout_rsfa_scale(int lane_id, int wave_id_m) {
    static_assert(Vec == T::VEC_SCALE_SF || Vec == T::VEC_SCALE_SF_PAIR);
    constexpr auto block_shape =
        opus::make_tuple(opus::number<T::T_M>{}, opus::number<T::W_M>{}, opus::number<Vec>{});
    constexpr auto block_dim = opus::make_tuple(opus::make_tuple(opus::p_dim{}, opus::p_dim{}),
                                                opus::make_tuple(opus::y_dim{}));
    return opus::make_layout<Vec>(
        block_shape,
        opus::unfold_x_stride(block_dim, block_shape,
                              opus::tuple{opus::number<T::VEC_SCALE_SF_PAIR>{}, 1_I}),
        opus::unfold_p_coord(block_dim, opus::tuple{wave_id_m, lane_id % T::W_M}));
}

template <class T, int Vec = T::VEC_SCALE_SF>
__device__ inline constexpr auto make_layout_rsfb_scale() {
    static_assert(Vec == T::VEC_SCALE_SF || Vec == T::VEC_SCALE_SF_PAIR);
    constexpr auto block_shape = opus::make_tuple(opus::number<Vec>{});
    constexpr auto block_dim = opus::make_tuple(opus::make_tuple(opus::y_dim{}));
    return opus::make_layout<Vec>(block_shape,
                                  opus::unfold_x_stride(block_dim, block_shape, opus::tuple{1_I}),
                                  opus::unfold_p_coord(block_dim, opus::tuple{}));
}

template <class T> __device__ inline auto make_layout_sa_scale(int wave_id_m, int wave_id_n) {
    constexpr int num_waves = T::BLOCK_SIZE / T::WARP_SIZE;
    constexpr auto sa_block_shape =
        opus::make_tuple(opus::number<T::smem_m_rep / num_waves>{}, opus::number<T::T_N>{},
                         opus::number<T::T_M>{}, opus::number<T::VEC_A>{});
    constexpr auto sa_block_dim =
        opus::make_tuple(opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
                         opus::make_tuple(opus::y_dim{}));
    return opus::make_layout(
        sa_block_shape,
        opus::unfold_x_stride(
            sa_block_dim, sa_block_shape,
            opus::tuple{opus::number<T::smem_linear_wave + T::smem_padding>{}, 1_I}),
        opus::unfold_p_coord(sa_block_dim, opus::tuple{wave_id_n, wave_id_m}));
}

template <class T>
__device__ inline constexpr auto make_layout_sb_scale(int wave_id_m, int wave_id_n) {
    constexpr int n_repeats = T::HALF_B_N / (T::NUM_WAVES * T::W_N);
    constexpr int k_per_wave_load = (T::WARP_SIZE / T::W_N) * T::VEC_B;
    constexpr int k_repeats = T::B_K / k_per_wave_load;
    constexpr auto sb_block_shape =
        opus::make_tuple(opus::number<T::T_N>{}, opus::number<T::T_M>{}, opus::number<n_repeats>{},
                         opus::number<k_repeats>{}, opus::number<T::VEC_B>{});
    constexpr auto sb_block_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}, opus::y_dim{}),
        opus::make_tuple(opus::y_dim{}));
    return opus::make_layout(
        sb_block_shape,
        opus::unfold_x_stride(
            sb_block_dim, sb_block_shape,
            opus::tuple{opus::number<T::smem_linear_wave + T::smem_padding>{}, 1_I}),
        opus::unfold_p_coord(sb_block_dim, opus::tuple{wave_id_n, wave_id_m}));
}

template <class T>
__device__ inline constexpr auto make_layout_ssfa_scale(int lane_id, int wave_id_m, int wave_id_n) {
    constexpr auto block_shape =
        opus::make_tuple(opus::number<T::SFA_PASSES_PER_CACHE_PANEL>{}, opus::number<T::T_N>{},
                         opus::number<T::T_M>{}, opus::number<T::SFA_K_COLUMNS_PER_WAVE>{},
                         opus::number<T::B_M / T::VEC_SF>{}, opus::number<T::VEC_SF>{});
    constexpr auto block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));
    const int lane_id_m = ((lane_id & 4) << 3) | ((lane_id & 3) << 1) | ((lane_id & 8) >> 3);
    return opus::make_layout<T::VEC_SF>(
        block_shape,
        opus::unfold_x_stride(block_dim, block_shape,
                              opus::tuple{opus::number<T::SFA_PANEL_PITCH>{}, 1_I}),
        opus::unfold_p_coord(block_dim,
                             opus::tuple{wave_id_n, wave_id_m, lane_id / T::W_M, lane_id_m}));
}

template <class T>
__device__ inline constexpr auto make_layout_ssfb_scale(int lane_id, int wave_id) {
    constexpr auto block_shape =
        opus::make_tuple(opus::number<T::SCALE_N_HALVES>{},
                         opus::number<T::SCALE_PANEL_K_CAPACITY>{}, opus::number<T::VEC_SF>{});
    constexpr auto block_dim =
        opus::make_tuple(opus::make_tuple(opus::p_dim{}), opus::make_tuple(opus::p_dim{}),
                         opus::make_tuple(opus::y_dim{}));
    return opus::make_layout<T::VEC_SF>(
        block_shape,
        opus::unfold_x_stride(block_dim, block_shape,
                              opus::tuple{opus::number<T::VEC_SF>{},
                                          opus::number<T::SCALE_N_HALVES * T::VEC_SF>{}, 1_I}),
        opus::unfold_p_coord(block_dim, opus::tuple{wave_id, lane_id}));
}

#endif

template <class T> struct accumulator_pin_traits {
    static constexpr bool enabled = false;
    static constexpr int agpr_base = 0;
};

template <class T, int N, int AGPR_BASE> struct pinned_accumulator_array : opus::array<T, N> {};

template <class T, int N, int AGPR_BASE>
struct accumulator_pin_traits<pinned_accumulator_array<T, N, AGPR_BASE>> {
    static constexpr bool enabled = true;
    static constexpr int agpr_base = AGPR_BASE;
};

#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
template <class T, int C_BEGIN, int COUNT, int EXPLICIT_AGPR_BASE = -1, int EXPLICIT_INDEX_BASE = 0,
          class MMA, class VA, class VB, class VC>
__device__ inline void mma_scale_group(MMA &mma, const VA &v_a, const VB &v_b, VC &v_c,
                                       unsigned int v_sfa, unsigned int v_sfb) {
    using tiled_mma = opus::remove_cvref_t<MMA>;
    using base_mma = typename tiled_mma::MMA;
    opus::static_for<COUNT>([&](auto i) {
        constexpr int c_index = C_BEGIN + decltype(i)::value;
        constexpr int m_repeat = c_index / T::E_N;
        constexpr int n_repeat = c_index % T::E_N;
        using pin_traits =
            opus_gemm_blockscale_mxscale_bpreshuffle_pin_detail::accumulator_pin_traits<
                opus::remove_cvref_t<VC>>;
        constexpr int c_dwords = sizeof(typename base_mma::vtype_c) / sizeof(opus::u32_t);
        if constexpr (EXPLICIT_AGPR_BASE >= 0) {
            [[clang::amdgpu_pin_agpr(EXPLICIT_AGPR_BASE +
                                     (c_index - EXPLICIT_INDEX_BASE) * c_dwords)]] v_c[c_index] =
                base_mma{}(v_a[m_repeat], v_b[n_repeat], v_c[c_index], static_cast<int>(v_sfa),
                           static_cast<int>(v_sfb), opus::number<m_repeat>{}, opus::number<0>{});
        } else if constexpr (pin_traits::enabled) {
            [[clang::amdgpu_pin_agpr(pin_traits::agpr_base + c_index * c_dwords)]] v_c[c_index] =
                base_mma{}(v_a[m_repeat], v_b[n_repeat], v_c[c_index], static_cast<int>(v_sfa),
                           static_cast<int>(v_sfb), opus::number<m_repeat>{}, opus::number<0>{});
        } else {
            v_c[c_index] =
                base_mma{}(v_a[m_repeat], v_b[n_repeat], v_c[c_index], static_cast<int>(v_sfa),
                           static_cast<int>(v_sfb), opus::number<m_repeat>{}, opus::number<0>{});
        }
    });
}

#endif

} // namespace opus_gemm_blockscale_mxscale_bpreshuffle_pin_detail

struct opus_gemm_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_panel_base_traits
    : opus_gemm_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_common {
    static constexpr int BLOCK_SIZE = 256;
    static constexpr int WARP_SIZE = 64;
    static constexpr int NUM_WAVES = BLOCK_SIZE / WARP_SIZE;
    static constexpr int MIN_WGS_PER_CU = 1;
    static constexpr int LOOP_UNROLL = 2;
    static constexpr bool RESET_SFA_BEFORE_LOAD = false;
    static constexpr int B_M = 256;
    static constexpr int B_N = 256;
    static constexpr int B_K = 128;
    static constexpr int T_M = 2;
    static constexpr int T_N = 2;
    static constexpr int T_K = 1;
    static constexpr int W_M = 16;
    static constexpr int W_N = 16;
    static constexpr int W_K = 128;
    static constexpr int HALF_B_M = B_M / 2;
    static constexpr int HALF_B_N = B_N / 2;
    static_assert(NUM_WAVES == 4);
    static_assert(NUM_WAVES == T_M * T_N * T_K);
    static_assert(HALF_B_M % (W_M * T_M) == 0);
    static_assert(HALF_B_N % (W_N * T_N) == 0);
    static_assert(B_K % (W_K * T_K) == 0);
    static constexpr int E_M = HALF_B_M / (W_M * T_M);
    static constexpr int E_N = HALF_B_N / (W_N * T_N);
    static constexpr int E_K = B_K / (W_K * T_K);
    static constexpr int VEC_A = 16;
    static constexpr int VEC_B = 16;
    static constexpr int VEC_C = 4;
    static constexpr int OUTPUT_TILES_PER_WG = 1;
    static constexpr int VEC_SCALE_A = 16;
    static constexpr int VEC_SCALE_SF = 4;
    static constexpr int VEC_SCALE_SF_PAIR = 8;
    static constexpr int GROUP_M = 1;
    static constexpr int GROUP_N = 128;
    static constexpr int GROUP_K = 128;
    static constexpr int MFMA_SCALE_GROUP_K = 32;
    static constexpr int SCALE_KGROUPS_PER_MFMA = W_K / MFMA_SCALE_GROUP_K;
    static constexpr int NUM_KGROUPS = B_K / MFMA_SCALE_GROUP_K;
    static constexpr int smem_linear_wave = WARP_SIZE * VEC_A;
    static constexpr int smem_sub = smem_linear_wave / B_K;
    static constexpr int smem_m_rep = HALF_B_M / smem_sub;
    static constexpr int smem_n_rep = HALF_B_N / smem_sub;
    static constexpr int smem_padding = 32;
    static constexpr int SCALE_N_HALVES = B_N / HALF_B_N;
    static_assert(NUM_KGROUPS == 4);
    static constexpr int a_buffer_load_insts = HALF_B_M * B_K / (BLOCK_SIZE * VEC_A);
    static constexpr int b_buffer_load_insts = HALF_B_N * B_K / (BLOCK_SIZE * VEC_B);
    static constexpr int a_ds_read_insts = E_M * W_M * W_K / (WARP_SIZE * VEC_A);
    static constexpr int b_ds_read_insts = E_N * W_N * W_K / (WARP_SIZE * VEC_B);
    static constexpr int SMEM_A_ELEMS = smem_m_rep * (smem_linear_wave + smem_padding) * 4;
    static constexpr int SMEM_B_ELEMS = smem_n_rep * (smem_linear_wave + smem_padding) * 4;
    static constexpr int SCALE_PANEL_K_CAPACITY = 64;
    static constexpr int SFA_K_COLUMNS_PER_WAVE = WARP_SIZE / W_M;
    static constexpr int SFA_K_COLUMNS_PER_PASS = NUM_WAVES * SFA_K_COLUMNS_PER_WAVE;
    static constexpr int SFA_PASSES_PER_CACHE_PANEL =
        SCALE_PANEL_K_CAPACITY / SFA_K_COLUMNS_PER_PASS;
    static constexpr int SFA_PANEL_PITCH = B_M;
    static constexpr int SFA_PANEL_BYTES = SFA_PANEL_PITCH * SCALE_PANEL_K_CAPACITY;
    static constexpr int SFB_PANEL_BYTES = SCALE_N_HALVES * SCALE_PANEL_K_CAPACITY * VEC_SF;
    static constexpr int LDS_BYTES =
        SMEM_A_ELEMS + SMEM_B_ELEMS + SFA_PANEL_BYTES + SFB_PANEL_BYTES;
    static_assert(E_M == 4 && E_N == 4 && E_K == 1);
    static_assert(VEC_SCALE_SF == E_M);
    static_assert(VEC_SCALE_SF_PAIR == 2 * VEC_SCALE_SF);
    static_assert(SFA_PASSES_PER_CACHE_PANEL * SFA_K_COLUMNS_PER_PASS == SCALE_PANEL_K_CAPACITY);
    static_assert(a_ds_read_insts == 8 && b_ds_read_insts == 8);
    static_assert(SFA_PANEL_BYTES == 16384 && SFB_PANEL_BYTES == 512);
    static_assert(LDS_BYTES == 152064 && LDS_BYTES <= 163840);
};

template <int Panel, int FixedK, bool Reset, int Unroll>
struct opus_gemm_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_panel_traits
    : opus_gemm_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_panel_base_traits {
    static constexpr int FIXED_K = FixedK, LOOP_UNROLL = Unroll;
    static constexpr bool RESET_SFA_BEFORE_LOAD = Reset;
    static constexpr int SCALE_PANEL_K_CAPACITY = Panel;
    static constexpr int SFA_PASSES_PER_CACHE_PANEL = Panel / SFA_K_COLUMNS_PER_PASS;
    static constexpr int SFA_PANEL_BYTES = B_M * Panel;
    static constexpr int SFB_PANEL_BYTES = SCALE_N_HALVES * Panel * VEC_SF;
    static constexpr int LDS_BYTES =
        SMEM_A_ELEMS + SMEM_B_ELEMS + SFA_PANEL_BYTES + SFB_PANEL_BYTES;
    static_assert(Panel == 16 || Panel == 32 || Panel == 64);
    static_assert(Unroll == 2 || Unroll == 4);
    static_assert(FixedK == 0 || (FixedK > 0 && FixedK % 128 == 0 && FixedK <= 16384));
    static_assert(LDS_BYTES <= 160 * 1024);
};


// Double-buffered LDS panel with matrix/scale overlap and staged output.
template<int BlockM = 192, int BlockN = 256, int ScalePanel = 128, int FixedK = 0, int WaveM = 4, int GroupM = 0, int CacheA = 0, int CacheB = 0, bool FragmentArrays = false, bool PinA = false, bool PinB = false,
          bool PinC = false, bool XcdOrder = false, bool BoundedOutput = false,
          int PublishM = 0>
struct opus_gemm_bpreshuffle_lds_panel_traits {
    static constexpr bool PIN_A = PinA, PIN_B = PinB, PIN_C = PinC;
    static constexpr bool FRAGMENT_ARRAYS = FragmentArrays || PinA || PinB || PinC;
    static constexpr bool XCD_ORDER = XcdOrder;
    static constexpr bool BOUNDED_OUTPUT = BoundedOutput;
    static constexpr int PUBLISH_M = PublishM;
    // Publish the next LDS tile before the last M repeat reloads B.
    static_assert(PublishM >= 0 && PublishM < BlockM / (WaveM * 16) - 1);
    // Pin both sides of loop-carried C values to keep MFMA C/D in place.
    static constexpr bool PIN_C_INPUT = true;
    static constexpr int BLOCK_SIZE = WaveM * 2 * 64;
    static constexpr int FIXED_K = FixedK;
    static constexpr int GROUPED_M = GroupM, CACHE_A = CacheA, CACHE_B = CacheB;
    static_assert(GroupM == 0 || GroupM == 4 || GroupM == 8);
    static_assert(CacheA == 0 || CacheA == 1 || CacheA == 17);
    static_assert(CacheB == 0 || CacheB == 1 || CacheB == 17);
    static constexpr int WARP_SIZE = 64;
    static constexpr int NUM_WAVES = BLOCK_SIZE / WARP_SIZE;
    static constexpr int MIN_WGS_PER_CU = 1;

    static constexpr int B_M = BlockM;
    static constexpr int B_N = BlockN;
    static constexpr int B_K = 128;

    static constexpr int T_M = WaveM;
    static constexpr int T_N = 2;
    static constexpr int T_K = 1;

    static constexpr int W_M = 16;
    static constexpr int W_N = 16;
    static constexpr int W_K = 128;

    static constexpr int HALF_B_M = B_M;
    static constexpr int HALF_B_N = B_N;

    static_assert(NUM_WAVES == T_M * T_N * T_K);
    static_assert(B_M % (T_M * W_M) == 0 && B_N % (T_N * W_N) == 0);
    static_assert((B_M == 64 || B_M == 96 || B_M == 128 || B_M == 192 || B_M == 256) && (B_N == 64 || B_N == 128 || B_N == 256));
    static_assert((WaveM == 2 || WaveM == 4) && T_N == 2);

    static constexpr int E_M = B_M / (T_M * W_M);
    static constexpr int E_N = B_N / (T_N * W_N);
    static constexpr int E_K = 1;
    static_assert(E_M >= 2 && B_N >= NUM_WAVES * 16);
    static constexpr int C_REGS = B_M * B_N / BLOCK_SIZE;

    static constexpr int VEC_A = 16;
    static constexpr int VEC_B = 16;
    static constexpr int VEC_C = 4;
    static constexpr int VEC_SCALE_A = 16;
    static constexpr int VEC_OUTPUT = 8;
    static constexpr int A_CHUNKS_PER_FRAGMENT = W_M * W_K / (WARP_SIZE * VEC_A);
    static constexpr int B_CHUNKS_PER_FRAGMENT = W_N * W_K / (WARP_SIZE * VEC_B);

    static constexpr int GROUP_M = 1;
    static constexpr int GROUP_N = 128;
    static constexpr int GROUP_K = 128;
    static constexpr int SCALE_N_HALVES = (B_N + GROUP_N - 1) / GROUP_N;

    static constexpr int SWIZZLE_GROUP_M = 4;
    static constexpr int SWIZZLE_GROUP_N = 8;
    static constexpr int SWIZZLE_MIN_N_TILES = 256;
    static constexpr int SWIZZLE_MAX_N = 2048;
    static constexpr int SWIZZLE_MIN_M = 4096;

    static constexpr int MIN_K = 128;
    static constexpr int MAX_K = ScalePanel * GROUP_K;
    static constexpr int LOOP_UNROLL = 2;

    static constexpr int smem_linear_wave = WARP_SIZE * VEC_A;
    static constexpr int smem_m_rep = B_M * B_K / smem_linear_wave;
    static constexpr int smem_n_rep = B_N * B_K / smem_linear_wave;
    static constexpr int smem_padding = 32;
    static constexpr int A_ROWS_PER_WAVE = WARP_SIZE / (B_K / VEC_A);

    static constexpr int A_STAGE = smem_m_rep * (smem_linear_wave + smem_padding);
    static constexpr int B_STAGE = smem_n_rep * (smem_linear_wave + smem_padding);
    static constexpr int NUM_STAGES = 2;
    static constexpr int MATRIX_LDS_BYTES = NUM_STAGES * (A_STAGE + B_STAGE);
    static constexpr int MATRIX_VMEM_INSTRUCTIONS =
        B_M * B_K / (BLOCK_SIZE * VEC_A) + B_N * B_K / (BLOCK_SIZE * VEC_B);
    static_assert(NUM_STAGES == 2);

    static constexpr int SCALE_PANEL = ScalePanel;
    static constexpr int SFA_BYTES = B_M * SCALE_PANEL;
    static constexpr int SFB_BYTES = SCALE_PANEL * sizeof(unsigned);
    static constexpr int SFA_K_PANEL = SCALE_PANEL < 32 ? SCALE_PANEL : 32;
    static constexpr int SFA_K_PASSES = SCALE_PANEL / SFA_K_PANEL;
    static constexpr int SFA_THREADS_PER_GROUP = BLOCK_SIZE / SFA_K_PANEL;
    static constexpr int SFA_K_COLUMNS_PER_WAVE = WARP_SIZE / SFA_THREADS_PER_GROUP;
    static constexpr int SFA_ROWS_PER_PASS = SFA_THREADS_PER_GROUP * VEC_SCALE_A;
    static constexpr int SFA_PASSES =
        (B_M + SFA_ROWS_PER_PASS - 1) / SFA_ROWS_PER_PASS;

    static_assert(BLOCK_SIZE % SCALE_PANEL == 0);
    static_assert(SFA_THREADS_PER_GROUP <= WARP_SIZE);
    static_assert(FIXED_K == 0 || (FIXED_K % B_K == 0 && FIXED_K <= MAX_K));
    static_assert(WARP_SIZE % SFA_THREADS_PER_GROUP == 0);
    static_assert(T_N * T_M * SFA_K_COLUMNS_PER_WAVE == SFA_K_PANEL);
    static_assert(SFA_K_PANEL * SFA_K_PASSES == SCALE_PANEL);
    static_assert(SFA_PASSES * SFA_ROWS_PER_PASS >= B_M);

    static constexpr int A_SCALE_PACKS = (E_M + 3) / 4;
    static constexpr int B_SCALE_PACKS = 1;
    static constexpr int LDS_BYTES = MATRIX_LDS_BYTES + SFA_BYTES + SFB_BYTES;

    static constexpr int C_LDS_ROW_STRIDE_ELEMS = B_N + 8;
    static constexpr int OUTPUT_PASSES = B_M * B_N / (BLOCK_SIZE * VEC_OUTPUT);
    static_assert(B_M * C_LDS_ROW_STRIDE_ELEMS * sizeof(opus::bf16_t) <= LDS_BYTES);
    static_assert(B_M * B_N % (BLOCK_SIZE * VEC_OUTPUT) == 0);
};


// Configuration for the regular pin-AGPR pipeline.
// The pipeline entry selects the schedule; these traits only describe its configuration.
enum class opus_gemm_blockscale_mxscale_bpreshuffle_pin_matrix_view {
    tile,
    matrix,
    remaining_rows
};
enum class opus_gemm_blockscale_mxscale_bpreshuffle_pin_output { lds, direct, stage_all };

template <class Base,
          opus_gemm_blockscale_mxscale_bpreshuffle_pin_matrix_view MatrixView =
              opus_gemm_blockscale_mxscale_bpreshuffle_pin_matrix_view::tile,
          opus_gemm_blockscale_mxscale_bpreshuffle_pin_output Output =
              opus_gemm_blockscale_mxscale_bpreshuffle_pin_output::lds,
          bool UnsignedThreadIndex = false, bool WideOutput = false, int GroupM = 0>
struct opus_gemm_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_traits : Base {
    static constexpr auto MATRIX_VIEW = MatrixView;
    static constexpr auto OUTPUT = Output;
    static constexpr bool UNSIGNED_THREAD_INDEX = UnsignedThreadIndex;
    static constexpr bool WIDE_OUTPUT = WideOutput;
    static constexpr int GROUPED_M = GroupM;
    static_assert(GroupM == 0 || GroupM == 4 || GroupM == 8 || GroupM == 16);
    using thread_index = std::conditional_t<UnsignedThreadIndex, unsigned, int>;
    static constexpr bool GUARD_SFB_LANE = requires { Base::RESET_SFA_BEFORE_LOAD; };
    static constexpr bool PAD_M = [] {
        if constexpr (requires { Base::PAD_M; })
            return Base::PAD_M;
        else
            return true;
    }();
    static constexpr bool RESET_SFA_BEFORE_LOAD = [] {
        if constexpr (requires { Base::RESET_SFA_BEFORE_LOAD; })
            return Base::RESET_SFA_BEFORE_LOAD;
        else
            return false;
    }();
    static constexpr int LOOP_UNROLL = [] {
        if constexpr (requires { Base::LOOP_UNROLL; })
            return Base::LOOP_UNROLL;
        else
            return 2;
    }();
    static constexpr int FIXED_K = [] {
        if constexpr (requires { Base::FIXED_K; })
            return Base::FIXED_K;
        else
            return 0;
    }();
    static constexpr int STATIC_M = 0, STATIC_N = 0, STATIC_K = 0;
    static constexpr bool STATIC_BATCH_STRIDES = false;
};

// Fixed candidates use contiguous matrix/scale strides derived from M/N/K.
// Other candidates continue to read the original runtime kargs fields.
template <class Base, int M, int N, int K, bool FixedBatchStrides = true>
struct opus_gemm_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_fixed_traits : Base {
    static constexpr int STATIC_M = M, STATIC_N = N, STATIC_K = K;
    static constexpr bool STATIC_BATCH_STRIDES = FixedBatchStrides;
};

#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
namespace opus_gemm_blockscale_mxscale_bpreshuffle_pin_detail {
static constexpr int A0_AGPR_BASE = 0, A1_AGPR_BASE = 32;
static constexpr int B0_AGPR_BASE = 64, B1_AGPR_BASE = 96;
static constexpr int STAGING_AGPR_BASE = 128, C01_AGPR_BASE = 148;
} // namespace opus_gemm_blockscale_mxscale_bpreshuffle_pin_detail

#endif

// Same geometry/resource contract as b_direct; a separate entry owns the overlap schedule.
template <int BM, int BN, int NB, int BD, int BC = 0, bool XCD = false, int FixedK = 7168,
          int WN = 4, int WGPerCU = 1, int ACache = 0, int StoreCache = 2, int Output = 0,
          int BlockK = 256, bool Contiguous = false, bool PinA = false, bool PinC = false,
          int GroupM = 0, int PublishM = 0>
struct opus_gemm_blockscale_mxscale_bpreshuffle_b_direct_overlap_traits
    : opus_gemm_blockscale_mxscale_bpreshuffle_b_direct_traits<
          BM, BN, NB, BD, BC, XCD, FixedK, WN, WGPerCU, ACache, StoreCache, Output,
          BlockK, Contiguous, PinA, PinC, GroupM> {
    static constexpr int PUBLISH_M = PublishM;
    static_assert(PublishM >= 0 && PublishM < BM / 16 - 1);
};
