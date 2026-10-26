// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "opus_gemm_layout_a8w8_blockscale_mxscale_bpreshuffle_ring_gfx950.cuh"
#include "opus_gemm_traits_a8w8_blockscale_mxscale_bpreshuffle_lds_ring_gfx950.cuh"

#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
namespace opus_gemm_bpreshuffle_lds_ring_layout {
using namespace opus;

// A layouts.
template <class T>
__device__ inline constexpr auto
make_layout_ga_scale(int lane_id, int wave_id_m, int group, int stride_a)
{
    constexpr auto ga_block_shape = opus::make_tuple(
        opus::number<T::B_M / (8 * T::T_M)>{}, 8_I, opus::number<T::T_M>{}, 8_I, 16_I);
    constexpr auto ga_block_dim =
        opus::make_tuple(opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
                         opus::make_tuple(opus::p_dim{}, opus::y_dim{}));
    return opus::make_layout<16>(
        ga_block_shape,
        opus::unfold_x_stride(ga_block_dim, ga_block_shape, opus::tuple{stride_a, 1_I}),
        opus::unfold_p_coord(ga_block_dim,
                             opus::tuple{group / T::T_M, lane_id / 8, wave_id_m, lane_id % 8}));
}

template <class T>
__device__ inline constexpr auto make_layout_sa_scale(int group, int stage)
{
    constexpr auto sa_block_shape =
        opus::make_tuple(opus::number<T::NUM_STAGES>{}, opus::number<T::B_M / 8>{}, 64_I, 16_I);
    constexpr auto sa_block_dim = opus::make_tuple(opus::make_tuple(opus::p_dim{}),
                                                   opus::make_tuple(opus::p_dim{}),
                                                   opus::make_tuple(opus::p_dim{}),
                                                   opus::make_tuple(opus::y_dim{}));
    return opus::make_layout<16>(
        sa_block_shape,
        opus::unfold_x_stride(sa_block_dim,
                              sa_block_shape,
                              opus::tuple{opus::number<T::A_STAGE>{},
                                          opus::number<T::smem_linear_wave + T::smem_padding>{},
                                          16_I,
                                          1_I}),
        opus::unfold_p_coord(sa_block_dim, opus::tuple{stage, group, 0_I}));
}

template <class T>
__device__ inline constexpr auto make_layout_ra_scale(int lane_id, int wave_id_m)
{
    constexpr auto ra_block_shape =
        opus::make_tuple(opus::number<T::E_M>{}, 2_I, opus::number<T::T_M>{}, 8_I, 2_I, 4_I, 16_I);
    constexpr auto ra_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));
    const int row = wave_id_m * 16 + lane_id % 16;
    return opus::make_layout<16>(
        ra_block_shape,
        opus::unfold_x_stride(
            ra_block_dim,
            ra_block_shape,
            opus::tuple{opus::number<T::smem_linear_wave + T::smem_padding>{}, 1_I}),
        opus::unfold_p_coord(
            ra_block_dim,
            opus::tuple{row / (8 * T::T_M), row % T::T_M, (row / T::T_M) % 8, lane_id / 16}));
}

// B layouts.
template <class T>
__device__ inline constexpr auto
make_layout_gb_scale(int lane_id, int wave_id_m, int wave_id_n, int stride_b)
{
    constexpr int lanes_n         = T::W_N;
    constexpr int lanes_k         = T::WARP_SIZE / lanes_n;
    constexpr int k_per_wave_load = lanes_k * T::VEC_B;
    constexpr int n_repeats       = T::HALF_B_N / (T::NUM_WAVES * lanes_n);
    constexpr int k_repeats       = T::B_K / k_per_wave_load;
    constexpr auto gb_block_shape = opus::make_tuple(opus::number<T::T_N>{},
                                                     opus::number<T::T_M>{},
                                                     opus::number<n_repeats>{},
                                                     opus::number<k_repeats>{},
                                                     opus::number<T::WARP_SIZE>{},
                                                     opus::number<T::VEC_B>{});
    constexpr auto gb_block_dim =
        opus::make_tuple(opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}),
                         opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));
    return opus::make_layout<T::VEC_B>(
        gb_block_shape,
        opus::unfold_x_stride(gb_block_dim, gb_block_shape, opus::tuple{T::W_N * stride_b, 1_I}),
        opus::unfold_p_coord(gb_block_dim, opus::tuple{wave_id_n, wave_id_m, lane_id}));
}

template <class T>
__device__ inline constexpr auto make_layout_sb_scale(int wave_id_m, int wave_id_n)
{
    constexpr int n_repeats       = T::HALF_B_N / (T::NUM_WAVES * T::W_N);
    constexpr int k_per_wave_load = (T::WARP_SIZE / T::W_N) * T::VEC_B;
    constexpr int k_repeats       = T::B_K / k_per_wave_load;
    constexpr auto sb_block_shape = opus::make_tuple(opus::number<T::T_N>{},
                                                     opus::number<T::T_M>{},
                                                     opus::number<n_repeats>{},
                                                     opus::number<k_repeats>{},
                                                     opus::number<T::VEC_B>{});
    constexpr auto sb_block_dim   = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}, opus::y_dim{}),
        opus::make_tuple(opus::y_dim{}));
    return opus::make_layout(
        sb_block_shape,
        opus::unfold_x_stride(
            sb_block_dim,
            sb_block_shape,
            opus::tuple{opus::number<T::smem_linear_wave + T::smem_padding>{}, 1_I}),
        opus::unfold_p_coord(sb_block_dim, opus::tuple{wave_id_n, wave_id_m}));
}

template <class T>
__device__ inline constexpr auto make_layout_rb_scale(int lane_id, int wave_id_n)
{
    constexpr int k_vectors       = T::W_N * T::W_K / (T::WARP_SIZE * T::VEC_B);
    constexpr auto rb_block_shape = opus::make_tuple(opus::number<T::E_N>{},
                                                     opus::number<T::T_N>{},
                                                     opus::number<T::E_K>{},
                                                     opus::number<k_vectors>{},
                                                     opus::number<T::WARP_SIZE>{},
                                                     opus::number<T::VEC_B>{});
    constexpr auto rb_block_dim   = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}, opus::y_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));
    return opus::make_layout<T::VEC_B>(
        rb_block_shape,
        opus::unfold_x_stride(
            rb_block_dim,
            rb_block_shape,
            opus::tuple{opus::number<T::smem_linear_wave + T::smem_padding>{}, 1_I}),
        opus::unfold_p_coord(rb_block_dim, opus::tuple{wave_id_n, lane_id}));
}

// Scale A layouts.
template <int Vec>
__device__ inline constexpr auto
make_layout_gsfa_fragment(int row, int column, int shape_m, int columns, int stride_sfa)
{
    const auto gsfa_block_shape =
        opus::make_tuple(columns, (shape_m + Vec - 1) / Vec, opus::number<Vec>{});
    constexpr auto gsfa_block_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}), opus::make_tuple(opus::p_dim{}, opus::y_dim{}));
    return opus::make_layout<Vec>(
               gsfa_block_shape,
               opus::unfold_x_stride(
                   gsfa_block_dim, gsfa_block_shape, opus::tuple{stride_sfa, 1_I}),
               opus::unfold_p_coord(gsfa_block_dim, opus::tuple{column, row / Vec})) +
           row % Vec;
}

template <int ScaleCopy>
__device__ inline constexpr auto
make_layout_rsfa(int row, int k_slice, int block_m, int columns, int base = 0)
{
    constexpr int spacing       = 4 / ScaleCopy;
    const auto rsfa_block_shape = opus::make_tuple(columns, block_m / 16, 16_I, 1_I);
    constexpr auto rsfa_block_dim =
        opus::make_tuple(opus::make_tuple(opus::p_dim{}),
                         opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}));
    return opus::make_layout<1>(
               rsfa_block_shape,
               opus::unfold_x_stride(rsfa_block_dim,
                                     rsfa_block_shape,
                                     opus::tuple{block_m * spacing, opus::number<spacing>{}}),
               opus::unfold_p_coord(rsfa_block_dim, opus::tuple{k_slice, row / 16, row % 16})) +
           base;
}

// Scale B layouts.
template <class T>
__device__ inline constexpr auto
make_layout_sfb_scale(int group_n, int k_slice, int columns, int stride)
{
    const auto sfb_block_shape   = opus::make_tuple(opus::number<T::B_GROUPS>{}, columns, 1_I);
    constexpr auto sfb_block_dim = opus::make_tuple(opus::make_tuple(opus::p_dim{}),
                                                    opus::make_tuple(opus::p_dim{}, opus::y_dim{}));
    return opus::make_layout<1>(
        sfb_block_shape,
        opus::unfold_x_stride(sfb_block_dim, sfb_block_shape, opus::tuple{stride, 1_I}),
        opus::unfold_p_coord(sfb_block_dim, opus::tuple{group_n, k_slice}));
}
} // namespace opus_gemm_bpreshuffle_lds_ring_layout

template <class Traits>
__device__ __forceinline__ void opus_gemm_blockscale_mxscale_bpreshuffle_lds_ring(
    opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 kargs)
{
    using T = opus::remove_cvref_t<Traits>;
    using namespace opus_gemm_blockscale_mxscale_bpreshuffle_ring_detail;
    namespace layout = opus_gemm_bpreshuffle_lds_ring_layout;
    using namespace opus;
    using D_C               = std::conditional_t<T::SPLIT_K == 1, bf16_t, float>;
    const int row           = block_id_y() * T::B_M;
    const int col           = block_id_x() * T::B_N;
    const int split         = block_id_z();
    const int lane_id       = thread_id_x() % 64;
    const int wave_id       = __builtin_amdgcn_readfirstlane(thread_id_x() / 64);
    const int wave_id_m     = wave_id % T::T_M;
    const int wave_id_n     = wave_id / T::T_M;
    const int total_loops   = (T::FIXED_K ? T::FIXED_K : kargs.k) / T::B_K;
    const int per_split     = total_loops / T::SPLIT_K;
    const int extra         = total_loops % T::SPLIT_K;
    const int loops         = per_split + (split < extra);
    const int tile_begin    = split * per_split + (split < extra ? split : extra);
    const int active_stages = loops < T::NUM_STAGES ? loops : T::NUM_STAGES;
    const int matrix_bytes  = active_stages * (T::A_STAGE + T::B_STAGE);
    const int scale_bytes   = T::B_M * loops;

    // Tile views, LDS ring, scale panels, and accumulator layout.
    auto g_a = make_gmem(
        reinterpret_cast<const fp8_t*>(kargs.ptr_a) + row * kargs.stride_a + tile_begin * 128,
        static_cast<unsigned>((kargs.m - row) * kargs.stride_a - tile_begin * 128));
    auto g_b   = make_gmem(reinterpret_cast<const fp8_t*>(kargs.ptr_b) + col * kargs.stride_b +
                         tile_begin * 2048);
    auto g_sfa = make_gmem(
        reinterpret_cast<const unsigned char*>(kargs.ptr_sfa) + row + tile_begin * kargs.stride_sfa,
        static_cast<unsigned>((total_loops - tile_begin - 1) * kargs.stride_sfa + kargs.m - row));
    auto g_sfb = make_gmem(reinterpret_cast<const unsigned char*>(kargs.ptr_sfb) +
                           (col / 128) * kargs.stride_sfb + tile_begin);
    auto g_c   = make_gmem(
        reinterpret_cast<D_C*>(kargs.ptr_c) +
            static_cast<int64_t>(split) * kargs.m * kargs.stride_c +
            static_cast<int64_t>(row) * kargs.stride_c + col,
        static_cast<unsigned>(
            (((kargs.m - row < T::B_M ? kargs.m - row : T::B_M) - 1) * kargs.stride_c + T::B_N) *
            sizeof(D_C)));
    extern __shared__ __attribute__((aligned(16))) char lds[];
    auto s_a   = make_smem(reinterpret_cast<fp8_t*>(lds));
    auto s_b   = make_smem(reinterpret_cast<fp8_t*>(lds + active_stages * T::A_STAGE));
    auto s_sfa = make_smem(reinterpret_cast<unsigned char*>(lds + matrix_bytes));
    auto s_sfb = make_smem(reinterpret_cast<unsigned char*>(lds + matrix_bytes + scale_bytes));
    const auto u_gb =
        layout::make_layout_gb_scale<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_b);
    const auto u_sb = layout::make_layout_sb_scale<T>(wave_id_m, wave_id_n);
    const auto u_rb = layout::make_layout_rb_scale<T>(lane_id, wave_id_n);
    const auto u_ra = layout::make_layout_ra_scale<T>(lane_id, wave_id_m);
    auto mma        = make_tiled_mma<fp8_t, fp8_t, fp32_t>(seq<T::E_M, T::E_N, 1>{},
                                                    seq<T::T_M, T::T_N, 1>{},
                                                    seq<16, 16, 128>{},
                                                    mfma_adaptor_swap_ab{});
    using BaseMMA   = typename decltype(mma)::MMA;
    array<typename BaseMMA::vtype_c, T::E_M * T::E_N> c{};
    auto prefetch = [&](auto slot, int kt) {
        constexpr int stage = decltype(slot)::value;
        static_for<(T::B_M / 8 + T::NUM_WAVES - 1) / T::NUM_WAVES>([&](auto pass) {
            const int group = decltype(pass)::value * T::NUM_WAVES + wave_id;
            if(group < T::B_M / 8)
            {
                const auto u_ga =
                    layout::make_layout_ga_scale<T>(lane_id, wave_id_m, group, kargs.stride_a);
                const auto u_sa = layout::make_layout_sa_scale<T>(group, stage);
                async_load<16>(g_a, s_a.ptr, u_ga, u_sa, kt * 128, 0_I, number<T::CACHECTL_A>{});
            }
        });
        async_load<16>(
            g_b, s_b.ptr, u_gb, u_sb + stage * T::B_STAGE, kt * 2048, 0_I, number<T::CACHECTL_B>{});
    };
    constexpr int distance = T::NUM_STAGES - 1;
    auto step              = [&](auto si, int kt, auto refill) {
        constexpr int stage = decltype(si)::value;
        if constexpr(decltype(refill)::value)
        {
            if(kt + distance <= loops)
            {
                constexpr int pending = (T::NUM_STAGES - 2) * T::VMEM_TILE;
                s_waitcnt_vmcnt(number<(pending < 63 ? pending : 63)>{});
            }
            else
            {
                s_waitcnt_vmcnt(0_I);
            }
            s_waitcnt_lgkmcnt(0_I);
            sync_threads();
        }
        const auto a = load<16>(s_a, u_ra + stage * T::A_STAGE);
        const auto b = load<16>(s_b, u_rb + stage * T::B_STAGE);
        array<unsigned, (T::E_M + 3) / 4> sfa{};
        array<unsigned, T::B_GROUPS> sfb;
        static_for<T::E_M>([&](auto mi) {
            constexpr int m   = decltype(mi)::value;
            const auto u_rsfa = layout::make_layout_rsfa<4>(
                (m * T::T_M + wave_id_m) * 16 + lane_id % 16, kt, T::B_M, loops);
            sfa[m / 4] |= static_cast<unsigned>(load<1>(s_sfa, u_rsfa)[0]) << ((m % 4) * 8);
        });
        static_for<T::B_GROUPS>([&](auto ni) {
            constexpr int n = decltype(ni)::value;
            sfb[n] = load<1>(s_sfb, layout::make_layout_sfb_scale<T>(n, kt, loops, loops))[0];
        });
        if constexpr(decltype(refill)::value)
        {
            if(kt + distance < loops)
                prefetch(number<(stage + distance) % T::NUM_STAGES>{}, kt + distance);
        }
        static_for<T::E_M>([&](auto mi) {
            constexpr int m = decltype(mi)::value;
            static_for<T::E_N>([&](auto ni) {
                constexpr int n  = decltype(ni)::value;
                constexpr int ci = m * T::E_N + n;
                constexpr int sg = n / (128 / (T::T_N * 16));
                c[ci]            = BaseMMA{}(slice(a, number<m * 32>{}, number<(m + 1) * 32>{}),
                                  slice(b, number<n * 32>{}, number<(n + 1) * 32>{}),
                                  c[ci],
                                  static_cast<int>(sfa[m / 4]),
                                  static_cast<int>(sfb[sg]),
                                  number<m % 4>{},
                                  0_I);
            });
        });
    };

    // ===== Prologue =====
    // Publish the scale panels and initial A/B ring slots.
    static_for<(T::MAX_SFA_BYTES + T::BLOCK_SIZE * 16 - 1) / (T::BLOCK_SIZE * 16)>([&](auto pass) {
        const int index = (thread_id_x() + decltype(pass)::value * T::BLOCK_SIZE) * 16;
        if(index < scale_bytes)
        {
            const int r       = index % T::B_M;
            const int kt      = index / T::B_M;
            const auto u_gsfa = layout::make_layout_gsfa_fragment<16>(
                r, kt, kargs.m - row, loops, kargs.stride_sfa);
            const auto u_ssfa = layout::make_layout_gsfa_fragment<16>(r, kt, T::B_M, loops, T::B_M);
            const auto value =
                opus_gemm_blockscale_mxscale_bpreshuffle_panel_half_panel::load_sfa_vector<T>(
                    g_sfa, layout_to_offsets<16>(u_gsfa)[0], kargs.m - row - r);
            store<16>(s_sfa, value, u_ssfa);
        }
    });
    if(thread_id_x() < T::B_GROUPS * loops)
    {
        const int index = thread_id_x();
        const auto u_gsfb =
            layout::make_layout_sfb_scale<T>(index / loops, index % loops, loops, kargs.stride_sfb);
        const auto u_ssfb =
            layout::make_layout_sfb_scale<T>(index / loops, index % loops, loops, loops);
        store<1>(s_sfb, load<1>(g_sfb, u_gsfb), u_ssfb);
    }
    const int initial_tiles = loops <= T::NUM_STAGES ? loops : distance;
    static_for<T::NUM_STAGES>([&](auto i) {
        if(decltype(i)::value < initial_tiles)
            prefetch(i, decltype(i)::value);
    });
    // Publish all scale bytes before any wave reads them. The short path has no refills.
    s_waitcnt_vmcnt(0_I);
    s_waitcnt_lgkmcnt(0_I);
    sync_threads();

    // ===== Main loop =====
    // The dynamic loop retains its original tail waits; the short path performs no refills.
    if(loops <= T::NUM_STAGES)
    {
        static_for<T::NUM_STAGES>([&](auto i) {
            if(decltype(i)::value < loops)
                step(i, decltype(i)::value, 0_I);
        });
    }
    else
    {
#pragma clang loop unroll(disable)
        for(int base = 0; base < loops; base += T::NUM_STAGES)
        {
            static_for<T::NUM_STAGES>([&](auto i) {
                if(base + decltype(i)::value < loops)
                    step(i, base + decltype(i)::value, 1_I);
            });
        }
    }

    // ===== Epilogue =====
    // All ring slots have retired. Partition C for FP32 partials or BF16 output.
    const auto pc =
        opus::make_tuple(wave_id_m, lane_id % mma.grpn_c, wave_id_n, lane_id / mma.grpn_c);
    const auto uc      = partition_layout_c<4>(mma, opus::make_tuple(kargs.stride_c, 1_I), pc);
    const auto offsets = layout_to_offsets<4>(uc);

    // ===== Write back =====
    if constexpr(T::SPLIT_K == 1 && T::OUTPUT == 2)
    {
        opus_gemm_blockscale_mxscale_bpreshuffle_register_prefetch_mfma16_output::
            store_mfma16_packed<T>(c, g_c, lane_id, wave_id_m, wave_id_n, kargs.stride_c);
    }
    else
    {
        static_for<T::E_M * T::E_N>([&](auto i) {
            store<4>(g_c,
                     cast<D_C>(c[decltype(i)::value]),
                     offsets[decltype(i)::value],
                     0,
                     number<T::STORE_CACHE>{});
        });
    }
}
#endif

template <class Traits>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, Traits::WG_PER_CU) void
opus_gemm_blockscale_mxscale_bpreshuffle_lds_ring_kernel(
    opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 kargs)
{
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
    opus_gemm_blockscale_mxscale_bpreshuffle_lds_ring<Traits>(kargs);
#endif
}

template <int SplitK, int Vec = 4, int Block = 128>
__global__ __launch_bounds__(Block, 2) void
opus_gemm_blockscale_mxscale_bpreshuffle_lds_ring_reduce(const float* __restrict__ workspace,
                                                         opus::bf16_t* __restrict__ out,
                                                         int elements)
{
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
    using namespace opus;
    const int index = (block_id_x() * Block + thread_id_x()) * Vec;
    if(index >= elements)
        return;
    vector_t<float, Vec> acc{};
    static_for<SplitK>([&](auto split) {
        auto source = make_gmem(workspace + static_cast<int64_t>(decltype(split)::value) * elements,
                                static_cast<unsigned>(elements * sizeof(float)));
        static_for<Vec / 4>([&](auto part) {
            constexpr int offset = decltype(part)::value * 4;
            const auto value     = load<4>(source, index + offset);
            static_for<4>([&](auto elem) {
                acc[offset + decltype(elem)::value] += value[decltype(elem)::value];
            });
        });
    });
    auto target             = make_gmem(out, static_cast<unsigned>(elements * sizeof(bf16_t)));
    constexpr int store_vec = Vec < 8 ? Vec : 8;
    static_for<Vec / store_vec>([&](auto part) {
        constexpr int offset = decltype(part)::value * store_vec;
        store<store_vec>(target,
                         cast<bf16_t>(slice(acc, number<offset>{}, number<offset + store_vec>{})),
                         index + offset);
    });
#endif
}
