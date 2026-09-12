#pragma once

#include <opus/hip_minimal.hpp>
#include <opus/opus.hpp>

#include "opus_gemm_traits_a8w8_mxscale_bpreshuffle_gfx950.cuh"

#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx950__)

#if defined(__HIP_DEVICE_COMPILE__) && !__has_attribute(amdgpu_pin_agpr)
#error "kid 9000 requires clang23 with the AMDGPU hard-pin patches"
#else

using opus::operator""_I;

template<class T, int MRepeat, class Mem, class Offsets, class V>
__device__ inline void load_a_mrepeat_scale(
    Mem& mem, const Offsets& offsets, V& dst) {
    static_assert(T::VEC_A == 16 && T::E_M == 4 && T::E_K == 1);
    static_assert(MRepeat >= 0 && MRepeat < T::E_M);
    // One 16x16x128 MFMA A operand has two 16-byte K64 pieces.
    opus::static_for<2>([&](auto j) {
        constexpr int i = MRepeat * 2 + decltype(j)::value;
        const auto value = mem.template load<T::VEC_A>(offsets[i]);
        opus::set_slice(
            dst, value,
            opus::number<i * T::VEC_A>{},
            opus::number<(i + 1) * T::VEC_A>{});
    });
}

template<class T, int Begin, int End, class Mem, class Offsets, class V>
__device__ inline void load_b_range_scale(
    Mem& mem, const Offsets& offsets, V& dst) {
    opus::static_for<End - Begin>([&](auto j) {
        constexpr int i = Begin + decltype(j)::value;
        auto value = mem.template load<T::VEC_B>(offsets[i]);
        opus::set_slice(
            dst,
            value,
            opus::number<i * T::VEC_B>{},
            opus::number<(i + 1) * T::VEC_B>{});
    });
}

__device__ inline void sched_barrier_pairs_scale() {
    __builtin_amdgcn_sched_group_barrier(0x08, 1, 0);
    __builtin_amdgcn_sched_group_barrier(0x02, 2, 0);
    __builtin_amdgcn_sched_group_barrier(0x08, 1, 0);
    __builtin_amdgcn_sched_group_barrier(0x02, 2, 0);
}

template<class T, int HALF_TILE_M, int M_REPEAT, int N_REPEAT, class MMA>
__device__ inline auto mma_scale_one(
    MMA& mma,
    const typename opus::remove_cvref_t<MMA>::vtype_a& v_a,
    const typename opus::remove_cvref_t<MMA>::vtype_b& v_b,
    typename opus::remove_cvref_t<MMA>::MMA::vtype_c v_c,
    unsigned int v_sfa,
    unsigned int v_sfb)
    -> typename opus::remove_cvref_t<MMA>::MMA::vtype_c {
    (void)mma;

    using tiled_mma = opus::remove_cvref_t<MMA>;
    using base_mma = typename tiled_mma::MMA;

    constexpr int a_len = tiled_mma::mma_a_len;
    constexpr int b_len = tiled_mma::mma_b_len;
    constexpr int a_offset = M_REPEAT * a_len;
    constexpr int b_offset = N_REPEAT * b_len;
    // A 128-row half has four M repeats.  Each half owns one 4-byte SFA pack,
    // so op_sel remains 0..3 instead of incorrectly trying to address bytes
    // 4..7 in one dword.
    constexpr int scale_op_sel_a = M_REPEAT;
    static_assert(HALF_TILE_M == 0 || HALF_TILE_M == 1);
    static_assert(T::E_M == 4);

    auto s_a = opus::slice(
        v_a, opus::number<a_offset>{}, opus::number<a_offset + a_len>{});
    auto s_b = opus::slice(
        v_b, opus::number<b_offset>{}, opus::number<b_offset + b_len>{});
    return base_mma{}(
        s_a,
        s_b,
        v_c,
        static_cast<int>(v_sfa),
        static_cast<int>(v_sfb),
        opus::number<scale_op_sel_a>{},
        opus::number<N_REPEAT>{});
}

#define MXFP8_MMA_PAIR(HALF_TILE_M, M_REPEAT, N_GROUP, VA, VB, C0, C1, SFA, SFB) \
    do {                                                                          \
        C0 = mma_scale_one<T, HALF_TILE_M, M_REPEAT, (N_GROUP) * 2>(              \
            mma, VA, VB, C0, (SFA)[HALF_TILE_M], SFB);                            \
        C1 = mma_scale_one<T, HALF_TILE_M, M_REPEAT, (N_GROUP) * 2 + 1>(          \
            mma, VA, VB, C1, (SFA)[HALF_TILE_M], SFB);                            \
    } while (false)

#define MXFP8_MMA_ONE(HALF_TILE_M, M_REPEAT, N_REPEAT, VA, VB, C, SFA, SFB) \
    do {                                                                     \
        C = mma_scale_one<T, HALF_TILE_M, M_REPEAT, N_REPEAT>(               \
            mma, VA, VB, C, (SFA)[HALF_TILE_M], SFB);                        \
    } while (false)

template<class T>
__device__ inline auto make_layout_ga_scale(int lane_id, int wave_id_m, int wave_id_n, int stride_a) {
    constexpr int threads_k = T::B_K / T::VEC_A; // 8
    constexpr int threads_m_per_block = T::BLOCK_SIZE / threads_k; // 32
    constexpr int threads_m_per_wave = T::WARP_SIZE / threads_k; // 8

    constexpr auto ga_block_shape = opus::make_tuple(
        opus::number<T::HALF_B_M / threads_m_per_block>{},
        opus::number<T::T_N>{},
        opus::number<threads_m_per_wave>{},
        opus::number<T::T_M>{},
        opus::number<threads_k>{},
        opus::number<T::VEC_A>{});
    constexpr auto ga_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout<T::VEC_A>(
        ga_block_shape,
        opus::unfold_x_stride(
            ga_block_dim, ga_block_shape, opus::tuple{stride_a, 1_I}),
        opus::unfold_p_coord(
            ga_block_dim,
            opus::tuple{wave_id_n, lane_id / threads_k, wave_id_m,
                        lane_id % threads_k}));
}

template<class T>
__device__ inline auto make_layout_sa_scale(int wave_id_m, int wave_id_n) {
    constexpr int num_waves = T::BLOCK_SIZE / T::WARP_SIZE;

    constexpr auto sa_block_shape = opus::make_tuple(
        opus::number<T::smem_m_rep / num_waves>{},
        opus::number<T::T_N>{},
        opus::number<T::T_M>{},
        opus::number<T::VEC_A>{});
    constexpr auto sa_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}));

    return opus::make_layout(
        sa_block_shape,
        opus::unfold_x_stride(
            sa_block_dim, sa_block_shape,
            opus::tuple{opus::number<T::smem_linear_wave + T::smem_padding>{}, 1_I}),
        opus::unfold_p_coord(sa_block_dim,
                             opus::tuple{wave_id_n, wave_id_m}));
}

template<class T>
__device__ inline auto make_layout_gb_scale(int lane_id, int wave_id_m, int wave_id_n, int stride_b) {
    // AITER (16,16) shuffle: a whole wave copies a contiguous N16 x K64
    // region. Its four requests cover two N16 groups and both K64 halves.
    static_assert(T::T_M == 2 && T::T_N == 2);
    static_assert(T::HALF_B_N == 128 && T::B_K == 128 && T::VEC_B == 16);
    const int producer_rep = wave_id_n * T::T_M + wave_id_m;
    return opus::make_layout<T::VEC_B>(
        opus::make_tuple(2_I, 2_I, opus::number<T::VEC_B>{}),
        opus::make_tuple(16 * stride_b, 1024_I, 1_I),
        opus::make_tuple(opus::underscore{}, opus::underscore{},
                         opus::underscore{})) +
        producer_rep * 32 * stride_b + lane_id * T::VEC_B;
}

template<class T>
__device__ inline auto make_layout_sb_scale(int wave_id_m, int wave_id_n) {
    constexpr int num_waves = T::BLOCK_SIZE / T::WARP_SIZE;

    constexpr auto sb_block_shape = opus::make_tuple(
        opus::number<T::smem_n_rep / num_waves>{},
        opus::number<T::T_N>{},
        opus::number<T::T_M>{},
        opus::number<T::VEC_B>{});
    constexpr auto sb_block_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}, opus::y_dim{}),
        opus::make_tuple(opus::y_dim{}));

    const int producer_rep = wave_id_n * T::T_M + wave_id_m;

    return opus::make_layout(
        sb_block_shape,
        opus::unfold_x_stride(
            sb_block_dim, sb_block_shape,
            opus::tuple{opus::number<T::smem_linear_wave + T::smem_padding>{}, 1_I}),
        opus::unfold_p_coord(sb_block_dim, opus::tuple{producer_rep}));
}

template<class T>
__device__ inline auto make_layout_ra_scale(int lane_id, int wave_id_m) {
    constexpr auto ra_block_shape = opus::make_tuple(
        opus::number<T::E_M>{},
        opus::number<T::T_M>{},
        opus::number<T::T_M>{},
        opus::number<T::W_M / T::T_M>{},
        opus::number<T::E_K>{},
        opus::number<T::W_M * T::W_K / T::WARP_SIZE / T::VEC_A>{},
        opus::number<T::WARP_SIZE / T::W_M>{},
        opus::number<T::VEC_A>{});

    constexpr auto ra_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}, opus::y_dim{},
                         opus::p_dim{}, opus::y_dim{}));

    const int lane_id_m = lane_id % T::W_M;
    return opus::make_layout<T::VEC_A>(
        ra_block_shape,
        opus::unfold_x_stride(
            ra_block_dim, ra_block_shape,
            opus::tuple{opus::number<T::smem_linear_wave + T::smem_padding>{}, 1_I}),
        opus::unfold_p_coord(
            ra_block_dim,
            opus::tuple{wave_id_m, lane_id_m % T::T_M,
                        lane_id_m / T::T_M, lane_id / T::W_M}));
}

template<class T>
__device__ inline auto make_layout_rb_scale(int lane_id, int wave_id_n) {
    static_assert(T::E_N == 4 && T::E_K == 1);
    static_assert(T::T_M == 2 && T::T_N == 2 && T::VEC_B == 16);
    constexpr int pitch = T::smem_linear_wave + T::smem_padding;
    // Consumer n=32*repeat+16*wave_n+(lane%16),
    //          k=64*k_piece+16*(lane/16)+byte.
    // Invert the coalesced producer while preserving operand flattening
    // (n_repeat, k_piece, byte) and the original four-adjacent-row SB map.
    return opus::make_layout<T::VEC_B>(
        opus::make_tuple(4_I, 2_I, opus::number<T::VEC_B>{}),
        opus::make_tuple(opus::number<4 * pitch>{},
                         opus::number<pitch>{}, 1_I),
        opus::make_tuple(opus::underscore{}, opus::underscore{},
                         opus::underscore{})) +
        2 * wave_id_n * pitch + lane_id * T::VEC_B;
}

template<class Traits>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, Traits::MIN_WGS_PER_CU) void gemm_a8w8_mxfp8_scale_kernel(opus_gemm_mxscale_bpreshuffle_kargs_gfx950 kargs) {
    using namespace opus;

    using T = opus::remove_cvref_t<Traits>;
    using D_A = opus::fp8_t;
    using D_B = opus::fp8_t;
    using D_C = std::conditional_t<T::OUTPUT_BF16, opus::bf16_t, opus::fp32_t>;
    using D_ACC = opus::fp32_t;
    using D_SF = unsigned char;
    using D_SF_PACK = unsigned int;

    const int wgid = block_id_x();
    const int num_tiles_n = opus_mxscale_bpreshuffle_ceil_div(kargs.n, T::B_N);
    const int block_m = wgid / num_tiles_n;
    const int block_n = wgid % num_tiles_n;
    const int row = block_m * T::B_M;
    const int col = block_n * T::B_N;

    const int batch_id = block_id_z();
    const int wave_id = __builtin_amdgcn_readfirstlane(thread_id_x() / T::WARP_SIZE);
    const int lane_id = thread_id_x() % T::WARP_SIZE;

    auto g_a = make_gmem(
        reinterpret_cast<const D_A*>(kargs.ptr_a) +
        batch_id * kargs.stride_a_batch + row * kargs.stride_a);
    auto g_b = make_gmem(
        reinterpret_cast<const D_B*>(kargs.ptr_b) +
        batch_id * kargs.stride_b_batch + col * kargs.stride_b);
    auto g_c = make_gmem(
        reinterpret_cast<D_C*>(kargs.ptr_c) +
        batch_id * kargs.stride_c_batch + row * kargs.stride_c + col);

    // Compact external scales are consumed once per reusable K panel. SFA has
    // physical [K/128,M] bytes; SFB has physical [N/128,K/128] bytes.
    auto g_sfa = make_gmem(
        reinterpret_cast<const D_SF*>(kargs.ptr_sfa) +
        batch_id * kargs.stride_sfa_batch + row,
        static_cast<unsigned int>(kargs.stride_sfa_batch - row));
    const int sfb_row_offset = (col / T::GROUP_N) * kargs.stride_sfb;
    auto g_sfb = make_gmem(
        reinterpret_cast<const D_SF*>(kargs.ptr_sfb) +
        batch_id * kargs.stride_sfb_batch + sfb_row_offset,
        static_cast<unsigned int>(kargs.stride_sfb_batch - sfb_row_offset));

    const int wave_id_m = wave_id % T::T_M;
    const int wave_id_n = wave_id / T::T_M;

    auto u_ga = make_layout_ga_scale<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_a);
    auto u_sa = make_layout_sa_scale<T>(wave_id_m, wave_id_n);
    auto u_ra = make_layout_ra_scale<T>(lane_id, wave_id_m);
    auto u_gb = make_layout_gb_scale<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_b);
    auto u_sb = make_layout_sb_scale<T>(wave_id_m, wave_id_n);
    auto u_rb = make_layout_rb_scale<T>(lane_id, wave_id_n);
    constexpr int smem_a_elem = T::smem_m_rep * (T::smem_linear_wave + T::smem_padding);
    constexpr int smem_b_elem = T::smem_n_rep * (T::smem_linear_wave + T::smem_padding);
    // One owned allocation lets the final output reuse both retired matrices.
    constexpr int matrix_lds_bytes = smem_a_elem * 4 * sizeof(D_A)
                                   + smem_b_elem * 4 * sizeof(D_B);
    alignas(16) __shared__ char smem_matrix[matrix_lds_bytes];
    auto s_a = make_smem(reinterpret_cast<D_A*>(smem_matrix));
    auto s_b = make_smem(reinterpret_cast<D_B*>(smem_matrix + smem_a_elem * 4 * sizeof(D_A)));
    auto s_c = make_smem(reinterpret_cast<D_C*>(smem_matrix));
    constexpr int c_lds_pitch = T::B_N + 8;
    static_assert(!T::OUTPUT_BF16 || T::B_M * c_lds_pitch * sizeof(D_C) == matrix_lds_bytes);

    // Matrix producer pairs own contiguous padded LDS rows.
    // A wave owns one M128 half and alternates row parity across requests.
    // Each B producer owns one N128 half in the consumer LDS layout.
    const bool produces_a = wave_id_n == 0;
    auto g_matrix = produces_a ? g_a : g_b;
    const int matrix_vector_offset = produces_a
        ? (wave_id_m * T::HALF_B_M + (lane_id / 8) * 2) * kargs.stride_a + (lane_id % 8) * 16
        : wave_id_m * T::HALF_B_N * kargs.stride_b + lane_id * 16;
    // Both public launch paths enforce stride_a == stride_b == k.
    const int matrix_pair_stride = 16 * kargs.stride_a;
    const int matrix_odd_stride = produces_a ? kargs.stride_a : 1024;
    const int matrix_k_stride = produces_a ? T::B_K : T::B_K * 16;
    constexpr int matrix_pitch = T::smem_linear_wave + T::smem_padding;
    auto* matrix_lds_base = (produces_a ? s_a.ptr : s_b.ptr) + wave_id_m * smem_a_elem;
    static_assert(smem_a_elem == smem_b_elem);
    // Materialize immutable address deltas once in VGPRs, leaving only
    // the shared K offset in the scalar issue path. IOFFSET shifts both ends.
    opus::vector_t<int, 16> matrix_addresses;
    opus::static_for<16>([&](auto issue_i) {
        constexpr int issue = decltype(issue_i)::value;
        constexpr int local = issue % 4;
        constexpr int immediate = local * matrix_pitch - (local ? 32 : 0);
        int address = matrix_vector_offset + (issue / 2) * matrix_pair_stride
            + (issue % 2) * matrix_odd_stride - immediate;
        asm volatile("" : "+v"(address));
        matrix_addresses[issue] = address;
    });
    auto prefetch_matrix_issue = [&](auto issue_i, int matrix_stage, int k_tile) {
        constexpr int issue = decltype(issue_i)::value;
        constexpr int local = issue % 4;
        constexpr int immediate = local * matrix_pitch - (local ? 32 : 0);
        auto* dst = matrix_lds_base + matrix_stage * 2 * smem_a_elem
            + (issue / 4) * 4 * matrix_pitch + (local ? 32 : 0);
        g_matrix.template async_load<16>(
            reinterpret_cast<void*>(reinterpret_cast<__UINTPTR_TYPE__>(dst)),
            matrix_addresses[issue], k_tile * matrix_k_stride,
            opus::number<immediate>{}, opus::number<0>{});
    };

    alignas(8) __shared__ char smem_sfa[T::SFA_PANEL_BYTES];
    alignas(8) __shared__ char smem_sfb[T::SFB_PANEL_BYTES];
    auto s_sfa = make_smem(reinterpret_cast<D_SF*>(smem_sfa));
    auto s_sfb = make_smem(reinterpret_cast<D_SF*>(smem_sfb));
    auto mma = make_tiled_mma<D_A, D_B, D_ACC>(
        seq<T::E_M, T::E_N, T::E_K>{},
        seq<T::T_M, T::T_N, T::T_K>{},
        seq<T::W_M, T::W_N, T::W_K>{},
        mfma_adaptor_swap_ab{});
    typename decltype(mma)::vtype_a v_a[2];
    typename decltype(mma)::vtype_b v_b;
    typename decltype(mma)::vtype_b v_b_second;
    using AccFragment = typename decltype(mma)::MMA::vtype_c;
    static_assert(decltype(mma)::mma_c_len == 4);

#define MXFP8_PIN_C(NAME, BASE) \
    __attribute__((amdgpu_pin_agpr(BASE))) AccFragment NAME = {}
    MXFP8_PIN_C(c00_0, 0);
    MXFP8_PIN_C(c00_1, 4);
    MXFP8_PIN_C(c00_2, 8);
    MXFP8_PIN_C(c00_3, 12);
    MXFP8_PIN_C(c00_4, 16);
    MXFP8_PIN_C(c00_5, 20);
    MXFP8_PIN_C(c00_6, 24);
    MXFP8_PIN_C(c00_7, 28);
    MXFP8_PIN_C(c00_8, 32);
    MXFP8_PIN_C(c00_9, 36);
    MXFP8_PIN_C(c00_10, 40);
    MXFP8_PIN_C(c00_11, 44);
    MXFP8_PIN_C(c00_12, 48);
    MXFP8_PIN_C(c00_13, 52);
    MXFP8_PIN_C(c00_14, 56);
    MXFP8_PIN_C(c00_15, 60);
    MXFP8_PIN_C(c01_0, 64);
    MXFP8_PIN_C(c01_1, 68);
    MXFP8_PIN_C(c01_2, 72);
    MXFP8_PIN_C(c01_3, 76);
    MXFP8_PIN_C(c01_4, 80);
    MXFP8_PIN_C(c01_5, 84);
    MXFP8_PIN_C(c01_6, 88);
    MXFP8_PIN_C(c01_7, 92);
    MXFP8_PIN_C(c01_8, 96);
    MXFP8_PIN_C(c01_9, 100);
    MXFP8_PIN_C(c01_10, 104);
    MXFP8_PIN_C(c01_11, 108);
    MXFP8_PIN_C(c01_12, 112);
    MXFP8_PIN_C(c01_13, 116);
    MXFP8_PIN_C(c01_14, 120);
    MXFP8_PIN_C(c01_15, 124);
    MXFP8_PIN_C(c10_0, 128);
    MXFP8_PIN_C(c10_1, 132);
    MXFP8_PIN_C(c10_2, 136);
    MXFP8_PIN_C(c10_3, 140);
    MXFP8_PIN_C(c10_4, 144);
    MXFP8_PIN_C(c10_5, 148);
    MXFP8_PIN_C(c10_6, 152);
    MXFP8_PIN_C(c10_7, 156);
    MXFP8_PIN_C(c10_8, 160);
    MXFP8_PIN_C(c10_9, 164);
    MXFP8_PIN_C(c10_10, 168);
    MXFP8_PIN_C(c10_11, 172);
    MXFP8_PIN_C(c10_12, 176);
    MXFP8_PIN_C(c10_13, 180);
    MXFP8_PIN_C(c10_14, 184);
    MXFP8_PIN_C(c10_15, 188);
    MXFP8_PIN_C(c11_0, 192);
    MXFP8_PIN_C(c11_1, 196);
    MXFP8_PIN_C(c11_2, 200);
    MXFP8_PIN_C(c11_3, 204);
    MXFP8_PIN_C(c11_4, 208);
    MXFP8_PIN_C(c11_5, 212);
    MXFP8_PIN_C(c11_6, 216);
    MXFP8_PIN_C(c11_7, 220);
    MXFP8_PIN_C(c11_8, 224);
    MXFP8_PIN_C(c11_9, 228);
    MXFP8_PIN_C(c11_10, 232);
    MXFP8_PIN_C(c11_11, 236);
    MXFP8_PIN_C(c11_12, 240);
    MXFP8_PIN_C(c11_13, 244);
    MXFP8_PIN_C(c11_14, 248);
    MXFP8_PIN_C(c11_15, 252);
#undef MXFP8_PIN_C

    D_SF_PACK v_sfa[2];
    D_SF_PACK v_sfb[T::SCALE_N_HALVES];
    opus::vector_t<D_SF_PACK, 2> v_sfb_next;
    D_SF_PACK v_sfa1_next;
    auto ga_offset = [&](int half_tile_m, int tile_k) { return half_tile_m * T::HALF_B_M * kargs.stride_a + tile_k * T::B_K; };
    auto gb_offset = [&](int half_tile_n, int tile_k) { return half_tile_n * T::HALF_B_N * kargs.stride_b + tile_k * T::B_K * 16; };
    auto sa_offset = [&](int stage, int half_tile_m) { return (stage * 2 + half_tile_m) * smem_a_elem; };
    auto sb_offset = [&](int stage, int half_tile_n) { return (stage * 2 + half_tile_n) * smem_b_elem; };
    // Runtime K controls every matrix iteration. The fixed panel size
    // is a cache capacity; panels are refreshed for arbitrarily long K.
    const int loops = opus_mxscale_bpreshuffle_ceil_div(kargs.k, T::B_K);
    static_assert(T::SCALE_PANEL_K_TILES == 64);
    constexpr int panel_mask = T::SCALE_PANEL_K_TILES - 1;

    opus::vector_t<D_SF, 16> panel_raw_a[4];
    auto load_sfa_panel = [&](int panel_begin) {
        const int tid = thread_id_x();
        const int group = tid & 15;
        const int owner_wave = group / 4;
        const int call = group % 4;
        const int source_row = (owner_wave / T::T_M) * T::HALF_B_M +
            (owner_wave % T::T_M) * T::W_M + call * T::T_M * T::W_M;
        opus::static_for<4>([&](auto pass_i) {
            constexpr int pass = decltype(pass_i)::value;
            const int column = panel_begin + tid / 16 + pass * 16;
            const int valid_column = column < loops ? column : loops - 1;
            panel_raw_a[pass] = load<16>(g_sfa,
                valid_column * kargs.stride_sfa + source_row);
        });
        __builtin_amdgcn_sched_barrier(0);
    };

    auto publish_sfa_panel = [&]() {
        // Raw vectors were issued together before this transpose. Each
        // quad owns four M repeats for one K column and one M128/wave-M
        // pair. Transpose each 4x4 byte block in registers, then publish
        // the final dword directly, with no intermediate LDS transpose.
        const int tid = thread_id_x();
        const int group = tid & 15;
        const int owner_wave = group / 4;
        const int call = group % 4;
        const unsigned int select_pair = (call & 1) ? 0x03070105u : 0x06020400u;
        const unsigned int select_quad = (call & 2) ? 0x03020706u : 0x05040100u;
        opus::static_for<4>([&](auto pass_i) {
            constexpr int pass = decltype(pass_i)::value;
            const int k_column = tid / 16 + pass * 16;
            const auto raw = panel_raw_a[pass];
            const auto words = __builtin_bit_cast(opus::vector_t<D_SF_PACK, 4>, raw);
            opus::static_for<4>([&](auto word_i) {
                constexpr int word = decltype(word_i)::value;
                const D_SF_PACK x = words[word];
                const D_SF_PACK adjacent = opus::mov_dpp(x, opus::number<0xb1>{});
                const D_SF_PACK pair = __builtin_amdgcn_perm(adjacent, x, select_pair);
                const D_SF_PACK opposite = opus::mov_dpp(pair, opus::number<0x4e>{});
                const D_SF_PACK packed = __builtin_amdgcn_perm(opposite, pair, select_quad);
                const int output_row = word * 4 + call;
                const int dst = k_column * T::SFA_PANEL_PITCH
                    + ((owner_wave % T::T_M) * T::W_M + output_row) * 8
                    + (owner_wave / T::T_M) * 4;
                store<4>(s_sfa, __builtin_bit_cast(opus::vector_t<D_SF, 4>, packed), dst);
            });
        });
    };

    auto load_sfa_dword = [&](int k_tile, int half_tile_m) {
        const int addr = (k_tile & panel_mask) * T::SFA_PANEL_PITCH
            + (wave_id_m * T::W_M + (lane_id & 15)) * 8 + half_tile_m * 4;
        return __builtin_bit_cast(D_SF_PACK, load<4>(s_sfa, addr));
    };

    auto load_sfb_dword = [&](int k_tile, int half_tile_n) {
        // The prologue already duplicated the compact byte across the
        // dword. Every lane reads the same address, preserving op_sel.
        return __builtin_bit_cast(D_SF_PACK,
            load<4>(s_sfb, ((k_tile & panel_mask) * T::SCALE_N_HALVES + half_tile_n) * 4));
    };

    auto refill_scale_panel = [&](int next_tile) {
        if ((next_tile & panel_mask) == 0) {
            // Current scales are resident in registers. Retire every reader
            // of the previous panel before overwriting the same allocation.
            s_waitcnt_vmcnt(0_I);
            s_waitcnt_lgkmcnt(0_I);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);
            load_sfa_panel(next_tile);
            D_SF_PACK raw_b = 0;
            if (wave_id < T::SCALE_N_HALVES) {
                const int global_column = next_tile + lane_id;
                const int valid_column = global_column < loops ? global_column : loops - 1;
                const auto raw = load<1>(g_sfb, valid_column, wave_id * kargs.stride_sfb);
                raw_b = static_cast<D_SF_PACK>(raw[0]);
            }
            s_waitcnt_vmcnt(0_I);
            publish_sfa_panel();
            if (wave_id < T::SCALE_N_HALVES) {
                const D_SF_PACK packed = raw_b * 0x01010101u;
                store<4>(s_sfb, __builtin_bit_cast(opus::vector_t<D_SF, 4>, packed),
                    (lane_id * T::SCALE_N_HALVES + wave_id) * 4);
            }
            s_waitcnt_lgkmcnt(0_I);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);
        }
    };

    async_load<T::VEC_A>(g_a, s_a.ptr, u_ga, u_sa + sa_offset(0, 0), ga_offset(0, 0));
    async_load<T::VEC_A>(g_a, s_a.ptr, u_ga, u_sa + sa_offset(0, 1), ga_offset(1, 0));
    __builtin_amdgcn_sched_barrier(0);

    // Prologue: preload the first bounded scale panel before the first barrier.
    load_sfa_panel(0);
    D_SF_PACK panel_sfb_raw = 0;
    if (wave_id < T::SCALE_N_HALVES) {
        const int valid_column = lane_id < loops ? lane_id : loops - 1;
        const auto raw = load<1>(g_sfb, valid_column, wave_id * kargs.stride_sfb);
        panel_sfb_raw = static_cast<D_SF_PACK>(raw[0]);
    }
    async_load<T::VEC_B>(g_b, s_b.ptr, u_gb, u_sb + sb_offset(0, 0), gb_offset(0, 0));
    async_load<T::VEC_B>(g_b, s_b.ptr, u_gb, u_sb + sb_offset(0, 1), gb_offset(1, 0));

    // B matrix requests can progress while the scale transpose publishes.
    publish_sfa_panel();
    s_waitcnt_vmcnt(0_I);

    if (wave_id < T::SCALE_N_HALVES) {
        const D_SF_PACK packed = panel_sfb_raw * 0x01010101u;
        store<4>(s_sfb,
            __builtin_bit_cast(opus::vector_t<D_SF, 4>, packed),
            (lane_id * T::SCALE_N_HALVES + wave_id) * 4);
    }

    s_waitcnt_lgkmcnt(0_I);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    int stage = 0;
    int tile = 0;

    // A single K128 block has no K1 producer.
    if (loops > 1) {
        // Both matrices preload K1. Main-loop publications release the matrix
        // stage for t+2 prefetches; the penultimate block issues no future request.
        // The main-loop producer mapping also covers this initial matrix stage.
        opus::static_for<8>([&](auto initial_issue) {
            using Issue = opus::number<decltype(initial_issue)::value + 0>;
            prefetch_matrix_issue(Issue{}, 1, 1);
        });

        // This guarded K1 block has B data ready by the first
        // main-loop VMEM drain; scales require no further global traffic.
        // The main-loop producer mapping also covers this initial matrix stage.
        opus::static_for<8>([&](auto initial_issue) {
            using Issue = opus::number<decltype(initial_issue)::value + 8>;
            prefetch_matrix_issue(Issue{}, 1, 1);
        });
        __builtin_amdgcn_sched_barrier(0);

    }

    // Seed the operand register roll. B-half0, SFB0, and both SFA
    // dwords for the first K tile are loaded once here.  SFA0 is later reused
    // in place, while SFA1 is carried in a dedicated next-tile VGPR.
    v_sfa[0] = load_sfa_dword(0, 0);
    v_sfa[1] = load_sfa_dword(0, 1);
    v_sfb[0] = load_sfb_dword(0, 0);
    v_sfb[1] = load_sfb_dword(0, 1);
    v_b = load<T::VEC_B>(s_b, u_rb + sb_offset(stage, 0));
    v_a[0] = load<T::VEC_A>(s_a, u_ra + sa_offset(stage, 0));
    // Seed every current-tile matrix half. Subsequent A1/B1 values roll
    // into dead operand slices near the end of the previous K iteration.
    v_a[1] = load<T::VEC_A>(s_a, u_ra + sa_offset(stage, 1));
    v_b_second = load<T::VEC_B>(s_b, u_rb + sb_offset(stage, 1));
    __builtin_amdgcn_sched_barrier(0);

    // Prefetch while two later K128 blocks exist; peel the penultimate block.
    // Future matrix addresses always use the complete runtime K index.
#pragma unroll 8
    for (tile = 0; tile + 2 < loops; ++tile) {
        refill_scale_panel(tile + 1);
        const int next_stage = stage ^ 1;
        const int future_tile = tile + 2;

        // The preceding K iteration prefetched both B scale halves.
        __builtin_amdgcn_sched_barrier(0);
        // Keep stage address expressions before the scheduling boundary;
        // removing them moves the final unrolled stage multiply between MFMAs.
        const auto u_sa_next_0 = u_sa + sa_offset(next_stage, 0);
        const auto u_sa_next_1 = u_sa + sa_offset(next_stage, 1);
        __builtin_amdgcn_sched_barrier(0);

        const auto rb0_next_offsets =
            opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 0));

        // All LDS reads here are compiler-visible intrinsics. Let LLVM's
        // waitcnt pass wait at actual operand consumers. The A/B data stage
        // handoff still has vmcnt(0)/lgkmcnt(0)/barrier below.
        __builtin_amdgcn_s_setprio(1);

        // A half 0 x B half 0 -> C[0][0] (128x128 wave quadrant).
        // After publication at MFMA5, all waves issue one t+2 request
        // behind each pair through MFMA38. Wave-uniform resources select
        // the A or B producer without role branches inside the K loop.
        MXFP8_MMA_PAIR(0, 0, 0, v_a[0], v_b, c00_0, c00_1, v_sfa, v_sfb[0]);
        sched_barrier_pairs_scale();

        MXFP8_MMA_PAIR(0, 0, 1, v_a[0], v_b, c00_2, c00_3, v_sfa, v_sfb[0]);
        sched_barrier_pairs_scale();

        MXFP8_MMA_ONE(0, 1, 0, v_a[0], v_b, c00_4, v_sfa, v_sfb[0]);
        __builtin_amdgcn_sched_barrier(0);

        // MFMA5: publish t+1 and release stage t between independent MFMAs.
        __builtin_amdgcn_s_setprio(0);
        s_waitcnt_vmcnt(0_I);
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_setprio(1);

        MXFP8_MMA_ONE(0, 1, 1, v_a[0], v_b, c00_5, v_sfa, v_sfb[0]);
        __builtin_amdgcn_sched_barrier(0);

        MXFP8_MMA_PAIR(0, 1, 1, v_a[0], v_b, c00_6, c00_7, v_sfa, v_sfb[0]);
        sched_barrier_pairs_scale();
        prefetch_matrix_issue(opus::number<0>{}, stage, future_tile);
        __builtin_amdgcn_sched_group_barrier(0x20, 1, 0);
        __builtin_amdgcn_sched_barrier(0);

        MXFP8_MMA_PAIR(
            0, 2, 0, v_a[0], v_b, c00_8, c00_9, v_sfa, v_sfb[0]);
        sched_barrier_pairs_scale();
        prefetch_matrix_issue(opus::number<1>{}, stage, future_tile);
        __builtin_amdgcn_sched_group_barrier(0x20, 1, 0);
        __builtin_amdgcn_sched_barrier(0);

        MXFP8_MMA_PAIR(
            0, 2, 1, v_a[0], v_b, c00_10, c00_11, v_sfa, v_sfb[0]);
        sched_barrier_pairs_scale();
        prefetch_matrix_issue(opus::number<2>{}, stage, future_tile);
        __builtin_amdgcn_sched_group_barrier(0x20, 1, 0);
        __builtin_amdgcn_sched_barrier(0);

        MXFP8_MMA_PAIR(
            0, 3, 0, v_a[0], v_b, c00_12, c00_13, v_sfa, v_sfb[0]);
        sched_barrier_pairs_scale();
        prefetch_matrix_issue(opus::number<3>{}, stage, future_tile);
        __builtin_amdgcn_sched_group_barrier(0x20, 1, 0);
        __builtin_amdgcn_sched_barrier(0);

        MXFP8_MMA_PAIR(
            0, 3, 1, v_a[0], v_b, c00_14, c00_15, v_sfa, v_sfb[0]);
        sched_barrier_pairs_scale();
        prefetch_matrix_issue(opus::number<4>{}, stage, future_tile);
        __builtin_amdgcn_sched_group_barrier(0x20, 1, 0);
        __builtin_amdgcn_sched_barrier(0);

        // A half 1 x B half 0 -> C[1][0] (128x128 wave quadrant).
        MXFP8_MMA_PAIR(
            1, 0, 0, v_a[1], v_b, c10_0, c10_1, v_sfa, v_sfb[0]);
        sched_barrier_pairs_scale();
        prefetch_matrix_issue(opus::number<5>{}, stage, future_tile);
        __builtin_amdgcn_sched_group_barrier(0x20, 1, 0);
        __builtin_amdgcn_sched_barrier(0);

        MXFP8_MMA_PAIR(
            1, 0, 1, v_a[1], v_b, c10_2, c10_3, v_sfa, v_sfb[0]);
        sched_barrier_pairs_scale();
        prefetch_matrix_issue(opus::number<6>{}, stage, future_tile);
        __builtin_amdgcn_sched_group_barrier(0x20, 1, 0);
        __builtin_amdgcn_sched_barrier(0);

        MXFP8_MMA_PAIR(
            1, 1, 0, v_a[1], v_b, c10_4, c10_5, v_sfa, v_sfb[0]);
        sched_barrier_pairs_scale();
        prefetch_matrix_issue(opus::number<7>{}, stage, future_tile);
        __builtin_amdgcn_sched_group_barrier(0x20, 1, 0);
        __builtin_amdgcn_sched_barrier(0);

        MXFP8_MMA_PAIR(
            1, 1, 1, v_a[1], v_b, c10_6, c10_7, v_sfa, v_sfb[0]);
        sched_barrier_pairs_scale();
        prefetch_matrix_issue(opus::number<8>{}, stage, future_tile);
        __builtin_amdgcn_sched_group_barrier(0x20, 1, 0);
        __builtin_amdgcn_sched_barrier(0);

        MXFP8_MMA_PAIR(
            1, 2, 0, v_a[1], v_b, c10_8, c10_9, v_sfa, v_sfb[0]);
        sched_barrier_pairs_scale();
        prefetch_matrix_issue(opus::number<9>{}, stage, future_tile);
        __builtin_amdgcn_sched_group_barrier(0x20, 1, 0);
        __builtin_amdgcn_sched_barrier(0);

        MXFP8_MMA_PAIR(
            1, 2, 1, v_a[1], v_b, c10_10, c10_11, v_sfa, v_sfb[0]);
        sched_barrier_pairs_scale();
        prefetch_matrix_issue(opus::number<10>{}, stage, future_tile);
        __builtin_amdgcn_sched_group_barrier(0x20, 1, 0);
        __builtin_amdgcn_sched_barrier(0);

        MXFP8_MMA_PAIR(
            1, 3, 0, v_a[1], v_b, c10_12, c10_13, v_sfa, v_sfb[0]);
        sched_barrier_pairs_scale();
        prefetch_matrix_issue(opus::number<11>{}, stage, future_tile);
        __builtin_amdgcn_sched_group_barrier(0x20, 1, 0);
        __builtin_amdgcn_sched_barrier(0);

        MXFP8_MMA_PAIR(
            1, 3, 1, v_a[1], v_b, c10_14, c10_15, v_sfa, v_sfb[0]);
        sched_barrier_pairs_scale();
        prefetch_matrix_issue(opus::number<12>{}, stage, future_tile);
        __builtin_amdgcn_sched_group_barrier(0x20, 1, 0);
        __builtin_amdgcn_sched_barrier(0);

        // Spread the dead B0 operand reads after MFMA32.
        load_b_range_scale<T, 0, 2>(s_b, rb0_next_offsets, v_b);
        __builtin_amdgcn_sched_group_barrier(0x100, 2, 0);
        __builtin_amdgcn_sched_barrier(0);

        const auto& v_b_n1 = v_b_second;

        // A half 0 x B half 1 -> C[0][1] (128x128 wave quadrant).
        MXFP8_MMA_PAIR(
            0, 0, 0, v_a[0], v_b_n1, c01_0, c01_1, v_sfa, v_sfb[1]);
        sched_barrier_pairs_scale();
        prefetch_matrix_issue(opus::number<13>{}, stage, future_tile);
        __builtin_amdgcn_sched_group_barrier(0x20, 1, 0);
        __builtin_amdgcn_sched_barrier(0);

        // Spread the dead B0 operand reads after MFMA34.
        load_b_range_scale<T, 2, 4>(s_b, rb0_next_offsets, v_b);
        __builtin_amdgcn_sched_group_barrier(0x100, 2, 0);
        __builtin_amdgcn_sched_barrier(0);

        MXFP8_MMA_PAIR(
            0, 0, 1, v_a[0], v_b_n1, c01_2, c01_3, v_sfa, v_sfb[1]);
        sched_barrier_pairs_scale();
        prefetch_matrix_issue(opus::number<14>{}, stage, future_tile);
        __builtin_amdgcn_sched_group_barrier(0x20, 1, 0);
        __builtin_amdgcn_sched_barrier(0);

        // Spread the dead B0 operand reads after MFMA36.
        load_b_range_scale<T, 4, 6>(s_b, rb0_next_offsets, v_b);
        __builtin_amdgcn_sched_group_barrier(0x100, 2, 0);
        __builtin_amdgcn_sched_barrier(0);

        // M0's four c01 consumers are complete, and the early publication
        // barrier has published next_stage. Roll only its two K64 pieces.
        const auto ra0_next_offsets =
            opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0));
        const auto ra1_next_offsets =
            opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 1));
        const auto rb1_next_offsets =
            opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 1));
        load_a_mrepeat_scale<T, 0>(s_a, ra0_next_offsets, v_a[0]);
        __builtin_amdgcn_sched_group_barrier(0x100, 2, 0);

        __builtin_amdgcn_s_setprio(1);

        // Continue C01 while next-tile operands roll into dead registers.
        MXFP8_MMA_ONE(
            0, 1, 0, v_a[0], v_b_n1, c01_4, v_sfa, v_sfb[1]);
        __builtin_amdgcn_sched_barrier(0);
        MXFP8_MMA_ONE(
            0, 1, 1, v_a[0], v_b_n1, c01_5, v_sfa, v_sfb[1]);
        __builtin_amdgcn_sched_barrier(0);
        prefetch_matrix_issue(opus::number<15>{}, stage, future_tile);
        __builtin_amdgcn_sched_group_barrier(0x20, 1, 0);
        __builtin_amdgcn_sched_barrier(0);

        // Spread the dead B0 operand reads after MFMA38.
        load_b_range_scale<T, 6, 8>(s_b, rb0_next_offsets, v_b);
        __builtin_amdgcn_sched_group_barrier(0x100, 2, 0);
        __builtin_amdgcn_sched_barrier(0);

        // Start panel SFA1 for tile t+1 in a dedicated VGPR; the remaining
        // ten c01 plus sixteen c11 MFMAs cover this immutable panel read.
        v_sfa1_next = load_sfa_dword(tile + 1, 1);

        // The current tile no longer consumes B-half0 or SFB0. Read both
        // next scale halves in one LDS b64; retain current SFB1 through c11.
        // B-half0 was read in four pairs at MFMA32/34/36/38; its values
        // are retained for the next K tile.
        v_sfb_next = __builtin_bit_cast(opus::vector_t<D_SF_PACK, 2>,
            load<8>(s_sfb, ((tile + 1) & panel_mask) * T::SCALE_N_HALVES * 4));
        v_sfb[0] = v_sfb_next[0];
        __builtin_amdgcn_sched_barrier(0);

        MXFP8_MMA_PAIR(
            0, 1, 1, v_a[0], v_b_n1, c01_6, c01_7, v_sfa, v_sfb[1]);
        sched_barrier_pairs_scale();

        // M1 is dead after c01_7; M2/M3 still use their current-tile slices.
        load_a_mrepeat_scale<T, 1>(s_a, ra0_next_offsets, v_a[0]);
        __builtin_amdgcn_sched_group_barrier(0x100, 2, 0);

        MXFP8_MMA_PAIR(
            0, 2, 0, v_a[0], v_b_n1, c01_8, c01_9, v_sfa, v_sfb[1]);
        sched_barrier_pairs_scale();

        MXFP8_MMA_PAIR(
            0, 2, 1, v_a[0], v_b_n1, c01_10, c01_11, v_sfa, v_sfb[1]);
        sched_barrier_pairs_scale();

        // M2 is dead after c01_11. Keep the old SFA0 pack through M3.
        load_a_mrepeat_scale<T, 2>(s_a, ra0_next_offsets, v_a[0]);
        __builtin_amdgcn_sched_group_barrier(0x100, 2, 0);

        MXFP8_MMA_PAIR(
            0, 3, 0, v_a[0], v_b_n1, c01_12, c01_13, v_sfa, v_sfb[1]);
        sched_barrier_pairs_scale();

        MXFP8_MMA_PAIR(
            0, 3, 1, v_a[0], v_b_n1, c01_14, c01_15, v_sfa, v_sfb[1]);
        sched_barrier_pairs_scale();

        // SFA0 is dead after the final c01 MFMA.  Reuse the same VGPR for the
        // next tile's SFA0 now, then hide this LDS read under all sixteen c11
        // MFMAs.  The true register anti-dependency prevents the read from
        // moving above the last current-tile SFA0 consumer.
        __builtin_amdgcn_sched_barrier(0);
        v_sfa[0] = load_sfa_dword(tile + 1, 0);
        // M3 is dead after c01_15. The earlier three slices already hold
        // tile t+1, so these last two K64 pieces complete the A0 roll.
        load_a_mrepeat_scale<T, 3>(s_a, ra0_next_offsets, v_a[0]);
        __builtin_amdgcn_sched_group_barrier(0x100, 2, 0);
        __builtin_amdgcn_sched_barrier(0);

        // C11: reorder independent accumulators and roll each operand after its last use.
        MXFP8_MMA_PAIR(1, 0, 0, v_a[1], v_b_n1, c11_0, c11_1, v_sfa, v_sfb[1]);
        sched_barrier_pairs_scale();

        MXFP8_MMA_PAIR(1, 0, 1, v_a[1], v_b_n1, c11_2, c11_3, v_sfa, v_sfb[1]);
        sched_barrier_pairs_scale();
        // MFMA52: these operand slices have no remaining current-K consumers.
        __builtin_amdgcn_sched_barrier(0);
        load_a_mrepeat_scale<T, 0>(s_a, ra1_next_offsets, v_a[1]);
        __builtin_amdgcn_sched_group_barrier(0x100, 2, 0);
        __builtin_amdgcn_sched_barrier(0);

        MXFP8_MMA_PAIR(1, 1, 0, v_a[1], v_b_n1, c11_4, c11_5, v_sfa, v_sfb[1]);
        sched_barrier_pairs_scale();

        MXFP8_MMA_PAIR(1, 1, 1, v_a[1], v_b_n1, c11_6, c11_7, v_sfa, v_sfb[1]);
        sched_barrier_pairs_scale();
        // MFMA56: these operand slices have no remaining current-K consumers.
        __builtin_amdgcn_sched_barrier(0);
        load_a_mrepeat_scale<T, 1>(s_a, ra1_next_offsets, v_a[1]);
        __builtin_amdgcn_sched_group_barrier(0x100, 2, 0);
        __builtin_amdgcn_sched_barrier(0);

        // MFMA56: preload the final operand slices into independent registers.
        auto a1_m3_next_0 = s_a.template load<16>(ra1_next_offsets[6]);
        auto a1_m3_next_1 = s_a.template load<16>(ra1_next_offsets[7]);
        __builtin_amdgcn_sched_group_barrier(0x100, 2, 0);
        __builtin_amdgcn_sched_barrier(0);

        MXFP8_MMA_ONE(1, 2, 0, v_a[1], v_b_n1, c11_8, v_sfa, v_sfb[1]);
        __builtin_amdgcn_sched_barrier(0);

        MXFP8_MMA_ONE(1, 3, 0, v_a[1], v_b_n1, c11_12, v_sfa, v_sfb[1]);
        __builtin_amdgcn_sched_barrier(0);
        // MFMA58: these operand slices have no remaining current-K consumers.
        __builtin_amdgcn_sched_barrier(0);
        load_b_range_scale<T, 0, 2>(s_b, rb1_next_offsets, v_b_second);
        __builtin_amdgcn_sched_group_barrier(0x100, 2, 0);
        __builtin_amdgcn_sched_barrier(0);

        MXFP8_MMA_ONE(1, 2, 1, v_a[1], v_b_n1, c11_9, v_sfa, v_sfb[1]);
        __builtin_amdgcn_sched_barrier(0);

        MXFP8_MMA_ONE(1, 3, 1, v_a[1], v_b_n1, c11_13, v_sfa, v_sfb[1]);
        __builtin_amdgcn_sched_barrier(0);
        // MFMA60: these operand slices have no remaining current-K consumers.
        __builtin_amdgcn_sched_barrier(0);
        load_b_range_scale<T, 2, 4>(s_b, rb1_next_offsets, v_b_second);
        __builtin_amdgcn_sched_group_barrier(0x100, 2, 0);
        __builtin_amdgcn_sched_barrier(0);

        MXFP8_MMA_ONE(1, 2, 2, v_a[1], v_b_n1, c11_10, v_sfa, v_sfb[1]);
        __builtin_amdgcn_sched_barrier(0);

        MXFP8_MMA_ONE(1, 3, 2, v_a[1], v_b_n1, c11_14, v_sfa, v_sfb[1]);
        __builtin_amdgcn_sched_barrier(0);
        // MFMA62: these operand slices have no remaining current-K consumers.
        __builtin_amdgcn_sched_barrier(0);
        load_b_range_scale<T, 4, 6>(s_b, rb1_next_offsets, v_b_second);
        __builtin_amdgcn_sched_group_barrier(0x100, 2, 0);
        __builtin_amdgcn_sched_barrier(0);

        MXFP8_MMA_ONE(1, 2, 3, v_a[1], v_b_n1, c11_11, v_sfa, v_sfb[1]);
        __builtin_amdgcn_sched_barrier(0);
        // MFMA63: these operand slices have no remaining current-K consumers.
        __builtin_amdgcn_sched_barrier(0);
        load_a_mrepeat_scale<T, 2>(s_a, ra1_next_offsets, v_a[1]);
        __builtin_amdgcn_sched_group_barrier(0x100, 2, 0);
        __builtin_amdgcn_sched_barrier(0);

        MXFP8_MMA_ONE(1, 3, 3, v_a[1], v_b_n1, c11_15, v_sfa, v_sfb[1]);
        __builtin_amdgcn_sched_barrier(0);
        // MFMA64: these operand slices have no remaining current-K consumers.
        __builtin_amdgcn_sched_barrier(0);
        opus::set_slice(v_a[1], a1_m3_next_0, opus::number<96>{}, opus::number<112>{});
        opus::set_slice(v_a[1], a1_m3_next_1, opus::number<112>{}, opus::number<128>{});
        load_b_range_scale<T, 6, 8>(s_b, rb1_next_offsets, v_b_second);
        __builtin_amdgcn_sched_group_barrier(0x100, 4, 0);
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_s_setprio(0);
        v_sfb[1] = v_sfb_next[1];
        v_sfa[1] = v_sfa1_next;
        stage = next_stage;
    }

    // Penultimate runtime K128 block: roll the final block with no
    // unused future matrix request. K128 skips this segment entirely.
    if (loops > 1) {
        refill_scale_panel(tile + 1);
        const int next_stage = stage ^ 1;

        // The preceding K iteration prefetched both B scale halves.
        __builtin_amdgcn_sched_barrier(0);
        // Keep stage address expressions before the scheduling boundary;
        // removing them moves the final unrolled stage multiply between MFMAs.
        const auto u_sa_next_0 = u_sa + sa_offset(next_stage, 0);
        const auto u_sa_next_1 = u_sa + sa_offset(next_stage, 1);
        __builtin_amdgcn_sched_barrier(0);

        const auto rb0_next_offsets =
            opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 0));

        // All LDS reads here are compiler-visible intrinsics. Let LLVM's
        // waitcnt pass wait at actual operand consumers. The A/B data stage
        // handoff still has vmcnt(0)/lgkmcnt(0)/barrier below.
        __builtin_amdgcn_s_setprio(1);

        // A half 0 x B half 0 -> C[0][0] (128x128 wave quadrant).
        // Publish the final block at MFMA5, then finish its predecessor
        // and roll its successor without any further global matrix requests.
        MXFP8_MMA_PAIR(0, 0, 0, v_a[0], v_b, c00_0, c00_1, v_sfa, v_sfb[0]);
        sched_barrier_pairs_scale();

        MXFP8_MMA_PAIR(0, 0, 1, v_a[0], v_b, c00_2, c00_3, v_sfa, v_sfb[0]);
        sched_barrier_pairs_scale();

        MXFP8_MMA_ONE(0, 1, 0, v_a[0], v_b, c00_4, v_sfa, v_sfb[0]);
        __builtin_amdgcn_sched_barrier(0);

        // MFMA5: publish t+1 and release stage t between independent MFMAs.
        __builtin_amdgcn_s_setprio(0);
        s_waitcnt_vmcnt(0_I);
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_setprio(1);

        MXFP8_MMA_ONE(0, 1, 1, v_a[0], v_b, c00_5, v_sfa, v_sfb[0]);
        __builtin_amdgcn_sched_barrier(0);

        MXFP8_MMA_PAIR(0, 1, 1, v_a[0], v_b, c00_6, c00_7, v_sfa, v_sfb[0]);
        sched_barrier_pairs_scale();

        MXFP8_MMA_PAIR(
            0, 2, 0, v_a[0], v_b, c00_8, c00_9, v_sfa, v_sfb[0]);
        sched_barrier_pairs_scale();

        MXFP8_MMA_PAIR(
            0, 2, 1, v_a[0], v_b, c00_10, c00_11, v_sfa, v_sfb[0]);
        sched_barrier_pairs_scale();

        MXFP8_MMA_PAIR(
            0, 3, 0, v_a[0], v_b, c00_12, c00_13, v_sfa, v_sfb[0]);
        sched_barrier_pairs_scale();

        MXFP8_MMA_PAIR(
            0, 3, 1, v_a[0], v_b, c00_14, c00_15, v_sfa, v_sfb[0]);
        sched_barrier_pairs_scale();

        // A half 1 x B half 0 -> C[1][0] (128x128 wave quadrant).
        MXFP8_MMA_PAIR(
            1, 0, 0, v_a[1], v_b, c10_0, c10_1, v_sfa, v_sfb[0]);
        sched_barrier_pairs_scale();

        MXFP8_MMA_PAIR(
            1, 0, 1, v_a[1], v_b, c10_2, c10_3, v_sfa, v_sfb[0]);
        sched_barrier_pairs_scale();

        MXFP8_MMA_PAIR(
            1, 1, 0, v_a[1], v_b, c10_4, c10_5, v_sfa, v_sfb[0]);
        sched_barrier_pairs_scale();

        MXFP8_MMA_PAIR(
            1, 1, 1, v_a[1], v_b, c10_6, c10_7, v_sfa, v_sfb[0]);
        sched_barrier_pairs_scale();

        MXFP8_MMA_PAIR(
            1, 2, 0, v_a[1], v_b, c10_8, c10_9, v_sfa, v_sfb[0]);
        sched_barrier_pairs_scale();

        MXFP8_MMA_PAIR(
            1, 2, 1, v_a[1], v_b, c10_10, c10_11, v_sfa, v_sfb[0]);
        sched_barrier_pairs_scale();

        MXFP8_MMA_PAIR(
            1, 3, 0, v_a[1], v_b, c10_12, c10_13, v_sfa, v_sfb[0]);
        sched_barrier_pairs_scale();

        MXFP8_MMA_PAIR(
            1, 3, 1, v_a[1], v_b, c10_14, c10_15, v_sfa, v_sfb[0]);
        sched_barrier_pairs_scale();

        // Spread the dead B0 operand reads after MFMA32.
        load_b_range_scale<T, 0, 2>(s_b, rb0_next_offsets, v_b);
        __builtin_amdgcn_sched_group_barrier(0x100, 2, 0);
        __builtin_amdgcn_sched_barrier(0);

        const auto& v_b_n1 = v_b_second;

        // A half 0 x B half 1 -> C[0][1] (128x128 wave quadrant).
        MXFP8_MMA_PAIR(
            0, 0, 0, v_a[0], v_b_n1, c01_0, c01_1, v_sfa, v_sfb[1]);
        sched_barrier_pairs_scale();

        // Spread the dead B0 operand reads after MFMA34.
        load_b_range_scale<T, 2, 4>(s_b, rb0_next_offsets, v_b);
        __builtin_amdgcn_sched_group_barrier(0x100, 2, 0);
        __builtin_amdgcn_sched_barrier(0);

        MXFP8_MMA_PAIR(
            0, 0, 1, v_a[0], v_b_n1, c01_2, c01_3, v_sfa, v_sfb[1]);
        sched_barrier_pairs_scale();

        // Spread the dead B0 operand reads after MFMA36.
        load_b_range_scale<T, 4, 6>(s_b, rb0_next_offsets, v_b);
        __builtin_amdgcn_sched_group_barrier(0x100, 2, 0);
        __builtin_amdgcn_sched_barrier(0);

        // M0's four c01 consumers are complete, and the early publication
        // barrier has published next_stage. Roll only its two K64 pieces.
        const auto ra0_next_offsets =
            opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0));
        const auto ra1_next_offsets =
            opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 1));
        const auto rb1_next_offsets =
            opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 1));
        load_a_mrepeat_scale<T, 0>(s_a, ra0_next_offsets, v_a[0]);
        __builtin_amdgcn_sched_group_barrier(0x100, 2, 0);

        __builtin_amdgcn_s_setprio(1);

        // Continue C01 while next-tile operands roll into dead registers.
        MXFP8_MMA_ONE(
            0, 1, 0, v_a[0], v_b_n1, c01_4, v_sfa, v_sfb[1]);
        __builtin_amdgcn_sched_barrier(0);
        MXFP8_MMA_ONE(
            0, 1, 1, v_a[0], v_b_n1, c01_5, v_sfa, v_sfb[1]);
        __builtin_amdgcn_sched_barrier(0);

        // Spread the dead B0 operand reads after MFMA38.
        load_b_range_scale<T, 6, 8>(s_b, rb0_next_offsets, v_b);
        __builtin_amdgcn_sched_group_barrier(0x100, 2, 0);
        __builtin_amdgcn_sched_barrier(0);

        // Start panel SFA1 for tile t+1 in a dedicated VGPR; the remaining
        // ten c01 plus sixteen c11 MFMAs cover this immutable panel read.
        v_sfa1_next = load_sfa_dword(tile + 1, 1);

        // The current tile no longer consumes B-half0 or SFB0. Read both
        // next scale halves in one LDS b64; retain current SFB1 through c11.
        // B-half0 was read in four pairs at MFMA32/34/36/38; its values
        // are retained for the next K tile.
        v_sfb_next = __builtin_bit_cast(opus::vector_t<D_SF_PACK, 2>,
            load<8>(s_sfb, ((tile + 1) & panel_mask) * T::SCALE_N_HALVES * 4));
        v_sfb[0] = v_sfb_next[0];
        __builtin_amdgcn_sched_barrier(0);

        MXFP8_MMA_PAIR(
            0, 1, 1, v_a[0], v_b_n1, c01_6, c01_7, v_sfa, v_sfb[1]);
        sched_barrier_pairs_scale();

        // M1 is dead after c01_7; M2/M3 still use their current-tile slices.
        load_a_mrepeat_scale<T, 1>(s_a, ra0_next_offsets, v_a[0]);
        __builtin_amdgcn_sched_group_barrier(0x100, 2, 0);

        MXFP8_MMA_PAIR(
            0, 2, 0, v_a[0], v_b_n1, c01_8, c01_9, v_sfa, v_sfb[1]);
        sched_barrier_pairs_scale();

        MXFP8_MMA_PAIR(
            0, 2, 1, v_a[0], v_b_n1, c01_10, c01_11, v_sfa, v_sfb[1]);
        sched_barrier_pairs_scale();

        // M2 is dead after c01_11. Keep the old SFA0 pack through M3.
        load_a_mrepeat_scale<T, 2>(s_a, ra0_next_offsets, v_a[0]);
        __builtin_amdgcn_sched_group_barrier(0x100, 2, 0);

        MXFP8_MMA_PAIR(
            0, 3, 0, v_a[0], v_b_n1, c01_12, c01_13, v_sfa, v_sfb[1]);
        sched_barrier_pairs_scale();

        MXFP8_MMA_PAIR(
            0, 3, 1, v_a[0], v_b_n1, c01_14, c01_15, v_sfa, v_sfb[1]);
        sched_barrier_pairs_scale();

        // SFA0 is dead after the final c01 MFMA.  Reuse the same VGPR for the
        // next tile's SFA0 now, then hide this LDS read under all sixteen c11
        // MFMAs.  The true register anti-dependency prevents the read from
        // moving above the last current-tile SFA0 consumer.
        __builtin_amdgcn_sched_barrier(0);
        v_sfa[0] = load_sfa_dword(tile + 1, 0);
        // M3 is dead after c01_15. The earlier three slices already hold
        // tile t+1, so these last two K64 pieces complete the A0 roll.
        load_a_mrepeat_scale<T, 3>(s_a, ra0_next_offsets, v_a[0]);
        __builtin_amdgcn_sched_group_barrier(0x100, 2, 0);
        __builtin_amdgcn_sched_barrier(0);

        // C11: reorder independent accumulators and roll each operand after its last use.
        MXFP8_MMA_PAIR(1, 0, 0, v_a[1], v_b_n1, c11_0, c11_1, v_sfa, v_sfb[1]);
        sched_barrier_pairs_scale();

        MXFP8_MMA_PAIR(1, 0, 1, v_a[1], v_b_n1, c11_2, c11_3, v_sfa, v_sfb[1]);
        sched_barrier_pairs_scale();
        // MFMA52: these operand slices have no remaining current-K consumers.
        __builtin_amdgcn_sched_barrier(0);
        load_a_mrepeat_scale<T, 0>(s_a, ra1_next_offsets, v_a[1]);
        __builtin_amdgcn_sched_group_barrier(0x100, 2, 0);
        __builtin_amdgcn_sched_barrier(0);

        MXFP8_MMA_PAIR(1, 1, 0, v_a[1], v_b_n1, c11_4, c11_5, v_sfa, v_sfb[1]);
        sched_barrier_pairs_scale();

        MXFP8_MMA_PAIR(1, 1, 1, v_a[1], v_b_n1, c11_6, c11_7, v_sfa, v_sfb[1]);
        sched_barrier_pairs_scale();
        // MFMA56: these operand slices have no remaining current-K consumers.
        __builtin_amdgcn_sched_barrier(0);
        load_a_mrepeat_scale<T, 1>(s_a, ra1_next_offsets, v_a[1]);
        __builtin_amdgcn_sched_group_barrier(0x100, 2, 0);
        __builtin_amdgcn_sched_barrier(0);

        // MFMA56: preload the final operand slices into independent registers.
        auto a1_m3_next_0 = s_a.template load<16>(ra1_next_offsets[6]);
        auto a1_m3_next_1 = s_a.template load<16>(ra1_next_offsets[7]);
        __builtin_amdgcn_sched_group_barrier(0x100, 2, 0);
        __builtin_amdgcn_sched_barrier(0);

        MXFP8_MMA_ONE(1, 2, 0, v_a[1], v_b_n1, c11_8, v_sfa, v_sfb[1]);
        __builtin_amdgcn_sched_barrier(0);

        MXFP8_MMA_ONE(1, 3, 0, v_a[1], v_b_n1, c11_12, v_sfa, v_sfb[1]);
        __builtin_amdgcn_sched_barrier(0);
        // MFMA58: these operand slices have no remaining current-K consumers.
        __builtin_amdgcn_sched_barrier(0);
        load_b_range_scale<T, 0, 2>(s_b, rb1_next_offsets, v_b_second);
        __builtin_amdgcn_sched_group_barrier(0x100, 2, 0);
        __builtin_amdgcn_sched_barrier(0);

        MXFP8_MMA_ONE(1, 2, 1, v_a[1], v_b_n1, c11_9, v_sfa, v_sfb[1]);
        __builtin_amdgcn_sched_barrier(0);

        MXFP8_MMA_ONE(1, 3, 1, v_a[1], v_b_n1, c11_13, v_sfa, v_sfb[1]);
        __builtin_amdgcn_sched_barrier(0);
        // MFMA60: these operand slices have no remaining current-K consumers.
        __builtin_amdgcn_sched_barrier(0);
        load_b_range_scale<T, 2, 4>(s_b, rb1_next_offsets, v_b_second);
        __builtin_amdgcn_sched_group_barrier(0x100, 2, 0);
        __builtin_amdgcn_sched_barrier(0);

        MXFP8_MMA_ONE(1, 2, 2, v_a[1], v_b_n1, c11_10, v_sfa, v_sfb[1]);
        __builtin_amdgcn_sched_barrier(0);

        MXFP8_MMA_ONE(1, 3, 2, v_a[1], v_b_n1, c11_14, v_sfa, v_sfb[1]);
        __builtin_amdgcn_sched_barrier(0);
        // MFMA62: these operand slices have no remaining current-K consumers.
        __builtin_amdgcn_sched_barrier(0);
        load_b_range_scale<T, 4, 6>(s_b, rb1_next_offsets, v_b_second);
        __builtin_amdgcn_sched_group_barrier(0x100, 2, 0);
        __builtin_amdgcn_sched_barrier(0);

        MXFP8_MMA_ONE(1, 2, 3, v_a[1], v_b_n1, c11_11, v_sfa, v_sfb[1]);
        __builtin_amdgcn_sched_barrier(0);
        // MFMA63: these operand slices have no remaining current-K consumers.
        __builtin_amdgcn_sched_barrier(0);
        load_a_mrepeat_scale<T, 2>(s_a, ra1_next_offsets, v_a[1]);
        __builtin_amdgcn_sched_group_barrier(0x100, 2, 0);
        __builtin_amdgcn_sched_barrier(0);

        MXFP8_MMA_ONE(1, 3, 3, v_a[1], v_b_n1, c11_15, v_sfa, v_sfb[1]);
        __builtin_amdgcn_sched_barrier(0);
        // MFMA64: these operand slices have no remaining current-K consumers.
        __builtin_amdgcn_sched_barrier(0);
        opus::set_slice(v_a[1], a1_m3_next_0, opus::number<96>{}, opus::number<112>{});
        opus::set_slice(v_a[1], a1_m3_next_1, opus::number<112>{}, opus::number<128>{});
        load_b_range_scale<T, 6, 8>(s_b, rb1_next_offsets, v_b_second);
        __builtin_amdgcn_sched_group_barrier(0x100, 4, 0);
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_s_setprio(0);
        v_sfb[1] = v_sfb_next[1];
        v_sfa[1] = v_sfa1_next;
        stage = next_stage;
    }

    // Consume the final resident tile and stage completed BF16 rows in LDS.
    if constexpr (T::OUTPUT_BF16) {
        s_waitcnt_vmcnt(0_I);
    }
    v_a[1] = load<T::VEC_A>(s_a, u_ra + sa_offset(stage, 1));

    s_waitcnt_lgkmcnt(0_I);
    if constexpr (T::OUTPUT_BF16) {
        // Every wave has completed all final-block matrix reads before any output write.
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
    }
    v_sfb[1] = load_sfb_dword(loops - 1, 1);

    __builtin_amdgcn_s_setprio(1);
    auto p_coord_c = opus::make_tuple(
        wave_id_m, lane_id % mma.grpn_c,
        wave_id_n, lane_id / mma.grpn_c);
    auto u_gc = partition_layout_c<T::VEC_C>(
        mma, opus::make_tuple(T::OUTPUT_BF16 ? c_lds_pitch : kargs.stride_c, 1_I), p_coord_c);

    auto c_offset = [&](int half_tile_m, int half_tile_n) {
        return half_tile_m * T::HALF_B_M * (T::OUTPUT_BF16 ? c_lds_pitch : kargs.stride_c) +
               half_tile_n * T::HALF_B_N;
    };

    // Each half covers all output rows and 128 contiguous columns.
    // The first half is published before its copy overlaps second-half MFMAs.
    auto copy_output_half = [&](int half_n, int copy_index) {
        const int linear = thread_id_x() * 8 + copy_index * T::BLOCK_SIZE * 8;
        const int output_row = linear / T::HALF_B_N;
        const int output_col = linear % T::HALF_B_N + half_n * T::HALF_B_N;
        const auto value = s_c.template load<8>(output_row * c_lds_pitch + output_col);
        g_c.template store<8>(value, output_row * kargs.stride_c + output_col,
                             0, opus::number<2>{});
    };

    const auto gc_offsets = opus::layout_to_offsets<T::VEC_C>(u_gc);
#define MXFP8_STORE_FRAGMENT(ACC, INDEX)                                   \
    do {                                                                        \
        if constexpr (T::OUTPUT_BF16) {                                         \
            s_c.template store<T::VEC_C>(cast<D_C>(ACC),                         \
                gc_offsets[INDEX] + soff);                                      \
        } else {                                                                \
            g_c.template store<T::VEC_C>(ACC, gc_offsets[INDEX], soff,            \
                opus::number<2>{});                                              \
        }                                                                       \
    } while (false)
// Store two native four-element MFMA fragments. BF16 goes to padded LDS;
// FP32 writes directly to C. No lane exchange is used by this final version.
#define MXFP8_STORE_TWO_FRAGMENTS(ACC0, ACC1, INDEX) \
    do { \
        MXFP8_STORE_FRAGMENT(ACC0, INDEX); \
        MXFP8_STORE_FRAGMENT(ACC1, (INDEX) + 1); \
    } while (false)
#define MXFP8_STORE_QUADRANT(PREFIX, HALF_M, HALF_N) \
    do { \
        const int soff = c_offset(HALF_M, HALF_N); \
        if constexpr (T::OUTPUT_BF16) { \
            MXFP8_STORE_TWO_FRAGMENTS(PREFIX##_0, PREFIX##_1, 0); \
            MXFP8_STORE_TWO_FRAGMENTS(PREFIX##_2, PREFIX##_3, 2); \
            MXFP8_STORE_TWO_FRAGMENTS(PREFIX##_4, PREFIX##_5, 4); \
            MXFP8_STORE_TWO_FRAGMENTS(PREFIX##_6, PREFIX##_7, 6); \
            MXFP8_STORE_TWO_FRAGMENTS(PREFIX##_8, PREFIX##_9, 8); \
            MXFP8_STORE_TWO_FRAGMENTS(PREFIX##_10, PREFIX##_11, 10); \
            MXFP8_STORE_TWO_FRAGMENTS(PREFIX##_12, PREFIX##_13, 12); \
            MXFP8_STORE_TWO_FRAGMENTS(PREFIX##_14, PREFIX##_15, 14); \
        } else { \
            MXFP8_STORE_FRAGMENT(PREFIX##_0, 0); \
            MXFP8_STORE_FRAGMENT(PREFIX##_1, 1); \
            MXFP8_STORE_FRAGMENT(PREFIX##_2, 2); \
            MXFP8_STORE_FRAGMENT(PREFIX##_3, 3); \
            MXFP8_STORE_FRAGMENT(PREFIX##_4, 4); \
            MXFP8_STORE_FRAGMENT(PREFIX##_5, 5); \
            MXFP8_STORE_FRAGMENT(PREFIX##_6, 6); \
            MXFP8_STORE_FRAGMENT(PREFIX##_7, 7); \
            MXFP8_STORE_FRAGMENT(PREFIX##_8, 8); \
            MXFP8_STORE_FRAGMENT(PREFIX##_9, 9); \
            MXFP8_STORE_FRAGMENT(PREFIX##_10, 10); \
            MXFP8_STORE_FRAGMENT(PREFIX##_11, 11); \
            MXFP8_STORE_FRAGMENT(PREFIX##_12, 12); \
            MXFP8_STORE_FRAGMENT(PREFIX##_13, 13); \
            MXFP8_STORE_FRAGMENT(PREFIX##_14, 14); \
            MXFP8_STORE_FRAGMENT(PREFIX##_15, 15); \
        } \
    } while (false)

    MXFP8_MMA_PAIR(0, 0, 0, v_a[0], v_b, c00_0, c00_1, v_sfa, v_sfb[0]);
    sched_barrier_pairs_scale();

    MXFP8_MMA_PAIR(0, 0, 1, v_a[0], v_b, c00_2, c00_3, v_sfa, v_sfb[0]);
    sched_barrier_pairs_scale();

    MXFP8_MMA_PAIR(0, 1, 0, v_a[0], v_b, c00_4, c00_5, v_sfa, v_sfb[0]);
    sched_barrier_pairs_scale();

    MXFP8_MMA_PAIR(0, 1, 1, v_a[0], v_b, c00_6, c00_7, v_sfa, v_sfb[0]);
    sched_barrier_pairs_scale();

    // Stage this completed output row while later independent MFMAs execute.
    if constexpr (T::OUTPUT_BF16) {
        const int soff = c_offset(0, 0);
        MXFP8_STORE_TWO_FRAGMENTS(c00_0, c00_1, 0);
        MXFP8_STORE_TWO_FRAGMENTS(c00_2, c00_3, 2);
    }

    MXFP8_MMA_PAIR(0, 2, 0, v_a[0], v_b, c00_8, c00_9, v_sfa, v_sfb[0]);
    sched_barrier_pairs_scale();

    MXFP8_MMA_PAIR(0, 2, 1, v_a[0], v_b, c00_10, c00_11, v_sfa, v_sfb[0]);
    sched_barrier_pairs_scale();

    // Stage this completed output row while later independent MFMAs execute.
    if constexpr (T::OUTPUT_BF16) {
        const int soff = c_offset(0, 0);
        MXFP8_STORE_TWO_FRAGMENTS(c00_4, c00_5, 4);
        MXFP8_STORE_TWO_FRAGMENTS(c00_6, c00_7, 6);
    }

    MXFP8_MMA_PAIR(0, 3, 0, v_a[0], v_b, c00_12, c00_13, v_sfa, v_sfb[0]);
    sched_barrier_pairs_scale();

    MXFP8_MMA_PAIR(0, 3, 1, v_a[0], v_b, c00_14, c00_15, v_sfa, v_sfb[0]);
    sched_barrier_pairs_scale();

    // Stage this completed output row while later independent MFMAs execute.
    if constexpr (T::OUTPUT_BF16) {
        const int soff = c_offset(0, 0);
        MXFP8_STORE_TWO_FRAGMENTS(c00_8, c00_9, 8);
        MXFP8_STORE_TWO_FRAGMENTS(c00_10, c00_11, 10);
    }

    MXFP8_MMA_PAIR(1, 0, 0, v_a[1], v_b, c10_0, c10_1, v_sfa, v_sfb[0]);
    sched_barrier_pairs_scale();

    MXFP8_MMA_PAIR(1, 0, 1, v_a[1], v_b, c10_2, c10_3, v_sfa, v_sfb[0]);
    sched_barrier_pairs_scale();

    // Stage this completed output row while later independent MFMAs execute.
    if constexpr (T::OUTPUT_BF16) {
        const int soff = c_offset(0, 0);
        MXFP8_STORE_TWO_FRAGMENTS(c00_12, c00_13, 12);
        MXFP8_STORE_TWO_FRAGMENTS(c00_14, c00_15, 14);
    }

    MXFP8_MMA_PAIR(1, 1, 0, v_a[1], v_b, c10_4, c10_5, v_sfa, v_sfb[0]);
    sched_barrier_pairs_scale();

    MXFP8_MMA_PAIR(1, 1, 1, v_a[1], v_b, c10_6, c10_7, v_sfa, v_sfb[0]);
    sched_barrier_pairs_scale();

    // Stage this completed output row while later independent MFMAs execute.
    if constexpr (T::OUTPUT_BF16) {
        const int soff = c_offset(1, 0);
        MXFP8_STORE_TWO_FRAGMENTS(c10_0, c10_1, 0);
        MXFP8_STORE_TWO_FRAGMENTS(c10_2, c10_3, 2);
    }

    MXFP8_MMA_PAIR(1, 2, 0, v_a[1], v_b, c10_8, c10_9, v_sfa, v_sfb[0]);
    sched_barrier_pairs_scale();

    MXFP8_MMA_PAIR(1, 2, 1, v_a[1], v_b, c10_10, c10_11, v_sfa, v_sfb[0]);
    sched_barrier_pairs_scale();

    // Stage this completed output row while later independent MFMAs execute.
    if constexpr (T::OUTPUT_BF16) {
        const int soff = c_offset(1, 0);
        MXFP8_STORE_TWO_FRAGMENTS(c10_4, c10_5, 4);
        MXFP8_STORE_TWO_FRAGMENTS(c10_6, c10_7, 6);
    }

    MXFP8_MMA_PAIR(1, 3, 0, v_a[1], v_b, c10_12, c10_13, v_sfa, v_sfb[0]);
    sched_barrier_pairs_scale();

    MXFP8_MMA_PAIR(1, 3, 1, v_a[1], v_b, c10_14, c10_15, v_sfa, v_sfb[0]);
    sched_barrier_pairs_scale();

    // Stage this completed output row while later independent MFMAs execute.
    if constexpr (T::OUTPUT_BF16) {
        const int soff = c_offset(1, 0);
        MXFP8_STORE_TWO_FRAGMENTS(c10_8, c10_9, 8);
        MXFP8_STORE_TWO_FRAGMENTS(c10_10, c10_11, 10);
    }

    // B-half0 is dead on the final K tile.  Start the direct AGPR stores for
    // both completed C quadrants before the B-half1 MFMAs occupy the XDL pipe.
    if constexpr (!T::OUTPUT_BF16) {
        MXFP8_STORE_QUADRANT(c00, 0, 0);
    }
    if constexpr (!T::OUTPUT_BF16) {
        MXFP8_STORE_QUADRANT(c10, 1, 0);
    }

    if constexpr (T::OUTPUT_BF16) {
        // Final B1 is resident from the operand seed or the penultimate block.
        v_b = v_b_second;
    } else {
        v_b = load<T::VEC_B>(s_b, u_rb + sb_offset(stage, 1));
    }

    MXFP8_MMA_PAIR(0, 0, 0, v_a[0], v_b, c01_0, c01_1, v_sfa, v_sfb[1]);
    sched_barrier_pairs_scale();

    MXFP8_MMA_PAIR(0, 0, 1, v_a[0], v_b, c01_2, c01_3, v_sfa, v_sfb[1]);
    sched_barrier_pairs_scale();

    // Stage this completed output row while later independent MFMAs execute.
    if constexpr (T::OUTPUT_BF16) {
        const int soff = c_offset(1, 0);
        MXFP8_STORE_TWO_FRAGMENTS(c10_12, c10_13, 12);
        MXFP8_STORE_TWO_FRAGMENTS(c10_14, c10_15, 14);
    }

    if constexpr (T::OUTPUT_BF16) {
        // All first-half rows are complete; the second half is disjoint.
        __builtin_amdgcn_sched_barrier(0);
        s_waitcnt_vmcnt(0_I);
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
        copy_output_half(0, 0);
        copy_output_half(0, 1);
        copy_output_half(0, 2);
        copy_output_half(0, 3);
        copy_output_half(0, 4);
        copy_output_half(0, 5);
        copy_output_half(0, 6);
        copy_output_half(0, 7);
        copy_output_half(0, 8);
        copy_output_half(0, 9);
        copy_output_half(0, 10);
        copy_output_half(0, 11);
        copy_output_half(0, 12);
        copy_output_half(0, 13);
        copy_output_half(0, 14);
        copy_output_half(0, 15);
    }

    MXFP8_MMA_PAIR(0, 1, 0, v_a[0], v_b, c01_4, c01_5, v_sfa, v_sfb[1]);
    sched_barrier_pairs_scale();

    MXFP8_MMA_PAIR(0, 1, 1, v_a[0], v_b, c01_6, c01_7, v_sfa, v_sfb[1]);
    sched_barrier_pairs_scale();

    // Stage this completed output row while later independent MFMAs execute.
    if constexpr (T::OUTPUT_BF16) {
        const int soff = c_offset(0, 1);
        MXFP8_STORE_TWO_FRAGMENTS(c01_0, c01_1, 0);
        MXFP8_STORE_TWO_FRAGMENTS(c01_2, c01_3, 2);
    }

    MXFP8_MMA_PAIR(0, 2, 0, v_a[0], v_b, c01_8, c01_9, v_sfa, v_sfb[1]);
    sched_barrier_pairs_scale();

    MXFP8_MMA_PAIR(0, 2, 1, v_a[0], v_b, c01_10, c01_11, v_sfa, v_sfb[1]);
    sched_barrier_pairs_scale();

    // Stage this completed output row while later independent MFMAs execute.
    if constexpr (T::OUTPUT_BF16) {
        const int soff = c_offset(0, 1);
        MXFP8_STORE_TWO_FRAGMENTS(c01_4, c01_5, 4);
        MXFP8_STORE_TWO_FRAGMENTS(c01_6, c01_7, 6);
    }

    MXFP8_MMA_PAIR(0, 3, 0, v_a[0], v_b, c01_12, c01_13, v_sfa, v_sfb[1]);
    sched_barrier_pairs_scale();

    MXFP8_MMA_PAIR(0, 3, 1, v_a[0], v_b, c01_14, c01_15, v_sfa, v_sfb[1]);
    sched_barrier_pairs_scale();

    // Stage this completed output row while later independent MFMAs execute.
    if constexpr (T::OUTPUT_BF16) {
        const int soff = c_offset(0, 1);
        MXFP8_STORE_TWO_FRAGMENTS(c01_8, c01_9, 8);
        MXFP8_STORE_TWO_FRAGMENTS(c01_10, c01_11, 10);
    }

    MXFP8_MMA_PAIR(1, 0, 0, v_a[1], v_b, c11_0, c11_1, v_sfa, v_sfb[1]);
    sched_barrier_pairs_scale();

    MXFP8_MMA_PAIR(1, 0, 1, v_a[1], v_b, c11_2, c11_3, v_sfa, v_sfb[1]);
    sched_barrier_pairs_scale();

    // Stage this completed output row while later independent MFMAs execute.
    if constexpr (T::OUTPUT_BF16) {
        const int soff = c_offset(0, 1);
        MXFP8_STORE_TWO_FRAGMENTS(c01_12, c01_13, 12);
        MXFP8_STORE_TWO_FRAGMENTS(c01_14, c01_15, 14);
    }

    MXFP8_MMA_PAIR(1, 1, 0, v_a[1], v_b, c11_4, c11_5, v_sfa, v_sfb[1]);
    sched_barrier_pairs_scale();

    MXFP8_MMA_PAIR(1, 1, 1, v_a[1], v_b, c11_6, c11_7, v_sfa, v_sfb[1]);
    sched_barrier_pairs_scale();

    // Stage this completed output row while later independent MFMAs execute.
    if constexpr (T::OUTPUT_BF16) {
        const int soff = c_offset(1, 1);
        MXFP8_STORE_TWO_FRAGMENTS(c11_0, c11_1, 0);
        MXFP8_STORE_TWO_FRAGMENTS(c11_2, c11_3, 2);
    }

    MXFP8_MMA_PAIR(1, 2, 0, v_a[1], v_b, c11_8, c11_9, v_sfa, v_sfb[1]);
    sched_barrier_pairs_scale();

    MXFP8_MMA_PAIR(1, 2, 1, v_a[1], v_b, c11_10, c11_11, v_sfa, v_sfb[1]);
    sched_barrier_pairs_scale();

    // Stage this completed output row while later independent MFMAs execute.
    if constexpr (T::OUTPUT_BF16) {
        const int soff = c_offset(1, 1);
        MXFP8_STORE_TWO_FRAGMENTS(c11_4, c11_5, 4);
        MXFP8_STORE_TWO_FRAGMENTS(c11_6, c11_7, 6);
    }

    MXFP8_MMA_PAIR(1, 3, 0, v_a[1], v_b, c11_12, c11_13, v_sfa, v_sfb[1]);
    sched_barrier_pairs_scale();

    MXFP8_MMA_PAIR(1, 3, 1, v_a[1], v_b, c11_14, c11_15, v_sfa, v_sfb[1]);
    sched_barrier_pairs_scale();

    // Stage this completed output row while later independent MFMAs execute.
    if constexpr (T::OUTPUT_BF16) {
        const int soff = c_offset(1, 1);
        MXFP8_STORE_TWO_FRAGMENTS(c11_8, c11_9, 8);
        MXFP8_STORE_TWO_FRAGMENTS(c11_10, c11_11, 10);
    }

    // Stage this completed output row while later independent MFMAs execute.
    if constexpr (T::OUTPUT_BF16) {
        const int soff = c_offset(1, 1);
        MXFP8_STORE_TWO_FRAGMENTS(c11_12, c11_13, 12);
        MXFP8_STORE_TWO_FRAGMENTS(c11_14, c11_15, 14);
    }
    __builtin_amdgcn_s_setprio(0);

    if constexpr (!T::OUTPUT_BF16) {
        MXFP8_STORE_QUADRANT(c01, 0, 1);
    }
    if constexpr (!T::OUTPUT_BF16) {
        MXFP8_STORE_QUADRANT(c11, 1, 1);
    }

    if constexpr (T::OUTPUT_BF16) {
        // The padded LDS rows distribute stores from the MFMA lane layout.
        // Retire first-half copies and publish the remaining output half.
        __builtin_amdgcn_sched_barrier(0);
        s_waitcnt_vmcnt(0_I);
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
        opus::static_for<T::B_M * T::HALF_B_N / (T::BLOCK_SIZE * 8)>([&](auto copy_i) {
            copy_output_half(1, decltype(copy_i)::value);
        });
    }
#undef MXFP8_STORE_QUADRANT
#undef MXFP8_STORE_TWO_FRAGMENTS
#undef MXFP8_STORE_FRAGMENT
#undef MXFP8_MMA_ONE
#undef MXFP8_MMA_PAIR
}

#endif // host pass or patched gfx950 device pass
#endif // host pass or gfx950 device pass
