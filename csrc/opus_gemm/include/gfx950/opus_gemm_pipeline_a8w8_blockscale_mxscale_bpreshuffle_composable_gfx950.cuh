// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "opus_gemm_layout_a8w8_blockscale_mxscale_bpreshuffle_ring_gfx950.cuh"
#include "opus_gemm_traits_a8w8_blockscale_mxscale_bpreshuffle_composable_gfx950.cuh"

struct opus_gemm_bpreshuffle_fused_split_kargs
{
    opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 args;
    opus::bf16_t* output;
    unsigned* counters;
};

#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
namespace opus_gemm_bpreshuffle_composable_layout {
using namespace opus;

// A layouts.
template <int BlockM, int BlockK, int BlockSize, class Copy>
__device__ inline constexpr auto make_layout_ga(Copy copy, int lane_id, int wave_id, int stride_a)
{
    constexpr int threads_k       = BlockK / 16;
    constexpr int rows_per_wave   = 64 / threads_k;
    constexpr int waves           = BlockSize / 64;
    constexpr int copies          = (BlockM * BlockK + BlockSize * 16 - 1) / (BlockSize * 16);
    constexpr auto ga_block_shape = opus::make_tuple(opus::number<copies>{},
                                                     opus::number<waves>{},
                                                     opus::number<rows_per_wave>{},
                                                     opus::number<threads_k>{},
                                                     16_I);
    constexpr auto ga_block_dim =
        opus::make_tuple(opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
                         opus::make_tuple(opus::p_dim{}, opus::y_dim{}));
    const auto logical_coord =
        opus::tuple{int(copy), wave_id, lane_id / threads_k, lane_id % threads_k};
    // XOR permutes K-vector ownership using the row coordinate. Compute that
    // coordinate from the same tile decomposition, independently of its pitch.
    const auto row_layout = opus::make_layout<-1>(
        ga_block_shape,
        opus::unfold_x_stride(ga_block_dim, ga_block_shape, opus::tuple{1_I, 0_I}),
        opus::unfold_p_coord(ga_block_dim, logical_coord));
    const int row_swizzle = row_layout(0_I) % threads_k;
    return opus::make_layout<16>(
        ga_block_shape,
        opus::unfold_x_stride(ga_block_dim, ga_block_shape, opus::tuple{stride_a, 1_I}),
        opus::unfold_p_coord(
            ga_block_dim,
            opus::tuple{
                int(copy), wave_id, lane_id / threads_k, (lane_id % threads_k) ^ row_swizzle}));
}

template <int BlockM, int BlockK, bool VectorOffsets = false>
__device__ inline constexpr auto make_layout_ra(int row, int lane_id, int k_slice, int base = 0)
{
    constexpr auto ra_block_shape = opus::make_tuple(
        opus::number<BlockM / 16>{}, 16_I, opus::number<BlockK / 128>{}, 2_I, 4_I, 16_I);
    constexpr auto ra_block_dim = opus::make_tuple(opus::make_tuple(opus::p_dim{}, opus::p_dim{}),
                                                   opus::make_tuple(opus::p_dim{}),
                                                   opus::make_tuple(opus::y_dim{}),
                                                   opus::make_tuple(opus::p_dim{}),
                                                   opus::make_tuple(opus::y_dim{}));
    const int swizzle           = row % (BlockK / 16);
    // XOR flips independent K-slice/lane coordinates. A flipped half axis
    // starts at half one and walks backwards, preserving MFMA operand order.
    const int half_stride = swizzle & 4 ? -64 : 64;
    auto layout =
        opus::make_layout<16>(
            ra_block_shape,
            opus::unfold_x_stride(
                ra_block_dim,
                ra_block_shape,
                opus::tuple{opus::number<BlockK>{}, 128_I, half_stride, 16_I, 1_I}),
            opus::unfold_p_coord(
                ra_block_dim,
                opus::tuple{
                    row / 16, row % 16, k_slice ^ (swizzle / 8), (lane_id / 16) ^ (swizzle % 4)})) +
        base + (swizzle & 4) * 16;
    if constexpr(VectorOffsets)
        return opus_gemm_blockscale_mxscale_bpreshuffle_ring_detail::vectorize_layout_offsets(
            layout);
    else
        return layout;
}

// B layouts.
template <int BlockN, int BlockK, int BlockSize, class Copy>
__device__ inline constexpr auto make_layout_gb(Copy copy, int lane_id, int wave_id, int stride_b)
{
    constexpr int waves_per_group = BlockK / 64;
    constexpr auto gb_block_shape =
        opus::make_tuple(opus::number<BlockN / 16>{}, opus::number<waves_per_group>{}, 64_I, 16_I);
    constexpr auto gb_block_dim =
        opus::make_tuple(opus::make_tuple(opus::p_dim{}),
                         opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}));
    const int copy_wave = int(copy) * (BlockSize / 64) + wave_id;
    return opus::make_layout<16>(
        gb_block_shape,
        opus::unfold_x_stride(gb_block_dim, gb_block_shape, opus::tuple{16 * stride_b, 1_I}),
        opus::unfold_p_coord(
            gb_block_dim,
            opus::tuple{copy_wave / waves_per_group, copy_wave % waves_per_group, lane_id}));
}

template <int BlockN>
__device__ inline constexpr auto
make_layout_rb(int group_n, int lane_id, int k_slice, int stride_b, int base = 0)
{
    const auto rb_block_shape =
        opus::make_tuple(opus::number<BlockN / 16>{}, stride_b / 128, 2_I, 64_I, 16_I);
    constexpr auto rb_block_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));
    return opus::make_layout<16>(
               rb_block_shape,
               opus::unfold_x_stride(rb_block_dim, rb_block_shape, opus::tuple{16 * stride_b, 1_I}),
               opus::unfold_p_coord(rb_block_dim, opus::tuple{group_n, k_slice, lane_id})) +
           base;
}

// Shared destinations layouts.
template <int Vec, int BlockSize, class Copy>
__device__ inline constexpr auto make_layout_dma(Copy copy, int wave_id, int base, int copies)
{
    constexpr int lane_stride = Vec < 4 ? 4 : Vec;
    const auto shared_block_shape =
        opus::make_tuple(copies, opus::number<BlockSize / 64>{}, 64_I, opus::number<Vec>{});
    constexpr auto shared_block_dim =
        opus::make_tuple(opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
                         opus::make_tuple(opus::y_dim{}));
    return opus::make_layout<Vec>(
               shared_block_shape,
               opus::unfold_x_stride(shared_block_dim,
                                     shared_block_shape,
                                     opus::tuple{opus::number<lane_stride>{}, 1_I}),
               opus::unfold_p_coord(shared_block_dim, opus::tuple{int(copy), wave_id, 0_I})) +
           base;
}

// Scale A layouts.
template <int Vec, int BlockM, int BlockSize, bool SimpleAddress = false, class Copy>
__device__ inline auto make_layout_gsfa(Copy copy,
                                        int tid,
                                        int row,
                                        int shape_m,
                                        int stride_sfa,
                                        int first_column,
                                        int columns,
                                        int total_columns)
{
    const auto gsfa_block_shape =
        opus::make_tuple(total_columns, (shape_m + Vec - 1) / Vec, opus::number<Vec>{});
    constexpr auto gsfa_block_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}), opus::make_tuple(opus::p_dim{}, opus::y_dim{}));
    const int copy_lane = int(copy) * BlockSize + tid;
    const int local_row = (copy_lane % (BlockM / Vec)) * Vec;
    const int column    = copy_lane / (BlockM / Vec);
    const bool valid    = column < columns && row + local_row < shape_m;
    int scale_column    = valid ? first_column + column : total_columns - 1;
    int scale_row       = valid ? row + local_row : shape_m;
    if constexpr(SimpleAddress)
    {
        const int bounded_column = column < columns ? column : columns - 1;
        scale_column             = first_column + bounded_column;
        scale_row                = row + local_row;
    }
    auto layout =
        opus::make_layout<Vec>(
            gsfa_block_shape,
            opus::unfold_x_stride(gsfa_block_dim, gsfa_block_shape, opus::tuple{stride_sfa, 1_I}),
            opus::unfold_p_coord(gsfa_block_dim, opus::tuple{scale_column, scale_row / Vec})) +
        scale_row % Vec;
    return opus_gemm_blockscale_mxscale_bpreshuffle_ring_detail::vectorize_layout_offsets(layout);
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
template <int BlockSize, class Copy>
__device__ inline constexpr auto make_layout_gsfb(Copy copy,
                                                  int tid,
                                                  int column,
                                                  int stride_sfb,
                                                  int first_column,
                                                  int columns,
                                                  int rows,
                                                  int total_groups)
{
    const auto gsfb_block_shape   = opus::make_tuple(total_groups, stride_sfb, 1_I);
    constexpr auto gsfb_block_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}), opus::make_tuple(opus::p_dim{}, opus::y_dim{}));
    const unsigned copy_lane = int(copy) * BlockSize + tid;
    const unsigned group_n   = copy_lane / columns < rows ? copy_lane / columns : rows - 1;
    return opus::make_layout<1>(
        gsfb_block_shape,
        opus::unfold_x_stride(gsfb_block_dim, gsfb_block_shape, opus::tuple{stride_sfb, 1_I}),
        opus::unfold_p_coord(
            gsfb_block_dim,
            opus::tuple{column / 128 + group_n, first_column + copy_lane % columns}));
}

template <int Groups>
__device__ inline constexpr auto
make_layout_rsfb(int group_n, int k_slice, int columns, int base = 0)
{
    const auto rsfb_block_shape   = opus::make_tuple(opus::number<Groups>{}, columns, 1_I);
    constexpr auto rsfb_block_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}), opus::make_tuple(opus::p_dim{}, opus::y_dim{}));
    return opus::make_layout<1>(
               rsfb_block_shape,
               opus::unfold_x_stride(
                   rsfb_block_dim, rsfb_block_shape, opus::tuple{columns * 4, 4_I}),
               opus::unfold_p_coord(rsfb_block_dim, opus::tuple{group_n, k_slice})) +
           base;
}

// C and split-K partials layouts.
template <bool Normal, int BlockM, int BlockN>
__device__ inline constexpr auto make_layout_rc(int row, int column, int lane_id, int stride_c)
{
    if constexpr(Normal)
    {
        constexpr auto rc_block_shape = opus::make_tuple(
            opus::number<BlockM / 16>{}, 4_I, 4_I, opus::number<BlockN / 16>{}, 16_I);
        constexpr auto rc_block_dim =
            opus::make_tuple(opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}),
                             opus::make_tuple(opus::p_dim{}, opus::p_dim{}));
        return opus::make_layout<1>(
            rc_block_shape,
            opus::unfold_x_stride(rc_block_dim, rc_block_shape, opus::tuple{stride_c, 1_I}),
            opus::unfold_p_coord(rc_block_dim,
                                 opus::tuple{row / 16, lane_id / 16, column / 16, lane_id % 16}));
    }
    else
    {
        constexpr auto rc_block_shape = opus::make_tuple(
            opus::number<BlockM / 16>{}, 16_I, opus::number<BlockN / 16>{}, 4_I, 4_I);
        constexpr auto rc_block_dim =
            opus::make_tuple(opus::make_tuple(opus::p_dim{}, opus::p_dim{}),
                             opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}));
        return opus::make_layout<4>(
            rc_block_shape,
            opus::unfold_x_stride(rc_block_dim, rc_block_shape, opus::tuple{stride_c, 1_I}),
            opus::unfold_p_coord(rc_block_dim,
                                 opus::tuple{row / 16, lane_id % 16, column / 16, lane_id / 16}));
    }
}

template <int Vec, int Rows, int Columns>
__device__ inline constexpr auto make_layout_matrix_vector(int row, int column, int stride)
{
    constexpr auto matrix_block_shape =
        opus::make_tuple(opus::number<Rows>{}, opus::number<Columns / Vec>{}, opus::number<Vec>{});
    constexpr auto matrix_block_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}), opus::make_tuple(opus::p_dim{}, opus::y_dim{}));
    return opus::make_layout<Vec>(
        matrix_block_shape,
        opus::unfold_x_stride(matrix_block_dim, matrix_block_shape, opus::tuple{stride, 1_I}),
        opus::unfold_p_coord(matrix_block_dim, opus::tuple{row, column / Vec}));
}

template <int BlockSize, int TileElements, int Splits>
__device__ inline constexpr auto make_layout_partial(int fragment, int tid, int split)
{
    constexpr auto partial_block_shape =
        opus::make_tuple(opus::number<Splits>{},
                         opus::number<TileElements / (BlockSize * 4)>{},
                         opus::number<BlockSize / 64>{},
                         64_I,
                         4_I);
    constexpr auto partial_block_dim = opus::make_tuple(opus::make_tuple(
        opus::p_dim{}, opus::p_dim{}, opus::p_dim{}, opus::p_dim{}, opus::y_dim{}));
    return opus::make_layout<4>(
        partial_block_shape,
        opus::unfold_x_stride(partial_block_dim, partial_block_shape, opus::tuple{1_I}),
        opus::unfold_p_coord(partial_block_dim, opus::tuple{split, fragment, tid / 64, tid % 64}));
}
} // namespace opus_gemm_bpreshuffle_composable_layout

template <class Traits>
__device__ __forceinline__ void opus_gemm_blockscale_mxscale_bpreshuffle_composable(
    opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 kargs,
    opus::bf16_t* final_output,
    unsigned* counters)
{
    using namespace opus;
    using namespace opus_gemm_blockscale_mxscale_bpreshuffle_ring_detail;
    namespace layout = opus_gemm_bpreshuffle_composable_layout;
    using T          = opus::remove_cvref_t<Traits>;
    using D_A        = opus::fp8_t;
    using D_B        = opus::fp8_t;
    using D_C        = opus::bf16_t;
    using D_ACC      = opus::fp32_t;
    using D_SF       = unsigned char;
    if constexpr(T::FIXED_N > 0)
        kargs.n = T::FIXED_N;
    kargs.k              = T::TOTAL_K;
    kargs.stride_a       = T::TOTAL_K;
    kargs.stride_b       = T::TOTAL_K;
    kargs.stride_c       = kargs.n;
    kargs.stride_sfa     = kargs.m;
    kargs.stride_sfb     = T::TOTAL_K / 128;
    const int m          = kargs.m;
    const int stride_c   = kargs.stride_c;
    const unsigned gm    = (kargs.m + T::B_M - 1) / T::B_M;
    const unsigned gn    = kargs.n / T::B_N;
    const unsigned tiles = gm * gn;
    unsigned tile = 0, split = 0;
    int row, col;
    constexpr bool m_first = [] {
        if constexpr(requires { T::M_FIRST; })
            return T::M_FIRST;
        else
            return false;
    }();
    if constexpr(m_first)
    {
        row = block_id_x() * T::B_M;
        col = block_id_y() * T::B_N;
    }
    else if constexpr(T::SPLIT_K == 1 && !T::XCD_ORDER)
    {
        row = block_id_y() * T::B_M;
        col = block_id_x() * T::B_N;
    }
    else
    {
        const unsigned pid = block_id_x();
        tile               = pid / (8 * T::SPLIT_K) * 8 + pid % 8;
        split              = (pid / 8) % T::SPLIT_K;
        if(tile >= tiles)
            return;
        row = (tile % gm) * T::B_M;
        col = (tile / gm) * T::B_N;
    }
    const int tid       = thread_id_x();
    const int lane_id   = tid % 64;
    const int lane_id_m = lane_id % 16;
    __builtin_assume(static_cast<unsigned>(tid) < T::BLOCK_SIZE);
    const int wave_id = __builtin_amdgcn_readfirstlane(tid / 64);
    __builtin_assume(static_cast<unsigned>(wave_id) < T::T_M * T::T_N);
    const int wave_id_m = wave_id / T::T_N;
    const int wave_id_n = wave_id % T::T_N;
    auto mfrag          = [&](auto mi) {
        return repeat_index<T::CONTIGUOUS>(wave_id_m, int(mi), T::E_M, T::T_M);
    };
    auto nfrag = [&](auto ni) {
        return repeat_index<T::CONTIGUOUS>(wave_id_n, int(ni), T::E_N, T::T_N);
    };
    const int valid_rows = m - row < T::B_M ? m - row : T::B_M;
    // Matrix global-memory views.
    auto g_a   = make_async_gmem(reinterpret_cast<const D_A*>(kargs.ptr_a) + row * kargs.stride_a,
                               valid_rows * kargs.stride_a);
    auto g_b   = make_async_gmem(reinterpret_cast<const D_B*>(kargs.ptr_b) + col * kargs.stride_b,
                               T::B_N * kargs.stride_b);
    auto g_sfa = make_async_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfa),
                                 (T::TOTAL_K / 128 - 1) * kargs.stride_sfa + m);
    auto g_sfb = make_async_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfb),
                                 (kargs.n / 128 - 1) * kargs.stride_sfb + T::TOTAL_K / 128);
    alignas(16) __shared__ D_SF smem[T::LDS_BYTES];
    auto s_ab = make_smem(reinterpret_cast<D_A*>(smem));
    auto s_sf = make_smem(smem);
    // MMA and register fragments.
    auto mma = make_tiled_mma<D_A, D_B, D_ACC>(
        seq<1, 1, 1>{},
        seq<1, 1, 1>{},
        seq<16, 16, 128>{},
        std::conditional_t<T::NORMAL, mfma_adaptor, mfma_adaptor_swap_ab>{});
    using BaseMMA    = typename decltype(mma)::MMA;
    using AFragments = array<array<typename BaseMMA::vtype_a, T::E_M>, T::KS>;
    using BFragments = array<array<typename BaseMMA::vtype_b, T::E_N>, T::KS>;
    struct Stage
    {
        AFragments a;
        BFragments b;
        array<array<unsigned, T::E_M>, T::KS> sa;
        array<array<unsigned, T::E_N>, T::KS> sb;
    };
    array<typename BaseMMA::vtype_c, T::E_M * T::E_N> c{};
    array<BFragments, T::B_SETS> b_regs;
    auto prefetch_b = [&](int t, auto phase) {
        if constexpr(T::B_DIRECT)
        {
            static_for<T::KS>([&](auto ki) {
                constexpr int ks = decltype(ki)::value;
                static_for<T::E_N>([&](auto ni) {
                    constexpr int n = decltype(ni)::value;
                    auto& frag      = b_regs[decltype(phase)::value][ks][n];
                    frag            = load<16>(
                        g_b,
                        layout::make_layout_rb<T::B_N>(
                            nfrag(ni), lane_id, ks, T::TOTAL_K, (split * T::K + t * T::B_K) * 16),
                        0,
                        number<T::B_CACHE>{});
                });
            });
            __builtin_amdgcn_sched_barrier(0);
        }
    };
    auto prefetch_scales = [&](int base, int kt) {
        static_for<T::SA_COPIES>([&](auto ii) {
            const auto u_gsfa =
                layout::make_layout_gsfa<T::SCALE_COPY, T::B_M, T::BLOCK_SIZE, T::SIMPLE_SCALE>(
                    ii,
                    tid,
                    row,
                    m,
                    kargs.stride_sfa,
                    split * T::SC + kt * T::KS,
                    T::SCALE_COLS,
                    T::TOTAL_K / 128);
            const auto u_ssfa = layout::make_layout_dma<T::SCALE_COPY, T::BLOCK_SIZE>(
                ii, wave_id, base + T::OFF_SA, T::SA_COPIES);
            async_load<T::SCALE_COPY>(g_sfa, s_sf.ptr, u_gsfa, u_ssfa);
        });
        static_for<T::SB_COPIES>([&](auto ii) {
            const auto u_gsfb = layout::make_layout_gsfb<T::BLOCK_SIZE>(ii,
                                                                        tid,
                                                                        col,
                                                                        kargs.stride_sfb,
                                                                        split * T::SC + kt * T::KS,
                                                                        T::SCALE_COLS,
                                                                        T::SB_ROWS,
                                                                        kargs.n / 128);
            const auto u_ssfb = layout::make_layout_dma<1, T::BLOCK_SIZE>(
                ii, wave_id, base + T::OFF_SB, T::SB_COPIES);
            async_load<1>(g_sfb, s_sf.ptr, u_gsfb, u_ssfb);
        });
    };
    auto prefetch = [&](int slot, int kt) {
        static_for<T::A_COPIES>([&](auto ii) {
            const auto u_ga = layout::make_layout_ga<T::B_M, T::B_K, T::BLOCK_SIZE>(
                ii, lane_id, wave_id, kargs.stride_a);
            const auto u_sa = layout::make_layout_dma<16, T::BLOCK_SIZE>(
                ii, wave_id, slot * T::STAGE_BYTES, T::A_COPIES);
            async_load<16>(g_a, s_ab.ptr, u_ga, u_sa, split * T::K + kt * T::B_K);
        });
        static_for<T::B_COPIES>([&](auto ii) {
            const auto u_gb = layout::make_layout_gb<T::B_N, T::B_K, T::BLOCK_SIZE>(
                ii, lane_id, wave_id, kargs.stride_b);
            const auto u_sb = layout::make_layout_dma<16, T::BLOCK_SIZE>(
                ii, wave_id, slot * T::STAGE_BYTES + T::A_BYTES, T::B_COPIES);
            async_load<16>(g_b,
                           s_ab.ptr,
                           u_gb,
                           u_sb,
                           (split * T::K + kt * T::B_K) * 16,
                           0_I,
                           number<T::B_CACHE>{});
        });
        if constexpr(T::PER_STAGE)
            prefetch_scales(slot * T::STAGE_BYTES, kt);
        commit_async_loads();
    };
    // Read all operands before issuing a later stage's DMA. The LLVM backend
    // must see those reads before that DMA to avoid an unnecessary vmcnt(0).
    auto read_stage = [&](int slot, int t, auto phase) {
        Stage f;
        auto& v_a   = f.a;
        auto& v_b   = f.b;
        auto& v_sfa = f.sa;
        auto& v_sfb = f.sb;
        static_for<T::KS>([&](auto ki) {
            constexpr int ks = decltype(ki)::value;
            static_for<T::E_N>([&](auto ni) {
                constexpr int n   = decltype(ni)::value;
                const unsigned bg = nfrag(ni) * 16 / 128;
                v_sfb[ks][n] =
                    load<1>(s_sf,
                            layout::make_layout_rsfb<T::SB_ROWS>(
                                bg,
                                (T::PER_STAGE ? 0 : t * T::KS) + ks,
                                T::SCALE_COLS,
                                (T::PER_STAGE ? slot * T::STAGE_BYTES : 0) + T::OFF_SB))[0];
            });
            // All M repeats differ by whole 16-row panels. Keep their static
            // displacement outside the dynamic address, including the swizzle.
            const int r0     = mfrag(0_I) * 16 + lane_id_m;
            const auto u_ra0 = layout::make_layout_ra<T::B_M, T::B_K, T::ADDRESS == 2>(
                r0, lane_id, ks, slot * T::STAGE_BYTES);
            static_for<T::E_M>([&](auto mi) {
                constexpr int m          = decltype(mi)::value;
                const int r              = mfrag(mi) * 16 + lane_id_m;
                constexpr unsigned delta = m * (T::CONTIGUOUS ? 1 : T::T_M) * 16;
                if constexpr(T::ADDRESS > 0 && delta % (T::B_K / 16) == 0)
                {
                    v_a[ks][m] = load<16>(s_ab, u_ra0 + number<delta * T::B_K>{});
                }
                else
                {
                    v_a[ks][m] = load<16>(s_ab,
                                          layout::make_layout_ra<T::B_M, T::B_K>(
                                              r, lane_id, ks, slot * T::STAGE_BYTES));
                }
                v_sfa[ks][m] =
                    load<1>(s_sf,
                            layout::make_layout_rsfa<T::SCALE_COPY>(
                                r,
                                (T::PER_STAGE ? 0 : t * T::KS) + ks,
                                T::B_M,
                                T::SCALE_COLS,
                                (T::PER_STAGE ? slot * T::STAGE_BYTES : 0) + T::OFF_SA))[0];
            });
            static_for<T::E_N>([&](auto ni) {
                constexpr int n = decltype(ni)::value;
                if constexpr(T::B_DIRECT)
                    v_b[ks][n] = b_regs[decltype(phase)::value][ks][n];
                else
                {
                    const int ng = nfrag(ni);
                    v_b[ks][n] =
                        load<16>(s_ab,
                                 layout::make_layout_rb<T::B_N>(
                                     ng, lane_id, ks, T::B_K, slot * T::STAGE_BYTES + T::A_BYTES));
                }
            });
        });
        return f;
    };
    auto compute = [&](const Stage& f) {
        static_for<T::KS>([&](auto ki) {
            constexpr int ks = decltype(ki)::value;
            static_for<T::E_M>([&](auto mi) {
                constexpr int m = decltype(mi)::value;
                static_for<T::E_N>([&](auto ni) {
                    constexpr int n   = decltype(ni)::value;
                    c[m * T::E_N + n] = BaseMMA{}(f.a[ks][m],
                                                  f.b[ks][n],
                                                  c[m * T::E_N + n],
                                                  int(f.sa[ks][m]),
                                                  int(f.sb[ks][n]),
                                                  0_I,
                                                  0_I);
                });
            });
        });
    };
    auto step = [&](int t, auto phase, auto pending, auto refill, auto refill_b) {
        if constexpr(T::ASYNC_GROUPS)
            wait_async_loads(pending);
        else
        {
            constexpr int wait =
                decltype(pending)::value * (T::MATRIX_ISSUES + T::SA_COPIES + T::SB_COPIES);
            s_waitcnt_vmcnt(number<(wait > 63 ? 63 : wait)>{});
        }
        sync_threads();
        if constexpr(T::B_DIRECT && decltype(refill_b)::value)
            prefetch_b(t + T::B_AHEAD, number<(decltype(phase)::value + T::B_AHEAD) % T::B_SETS>{});
        auto f = read_stage(t % T::STAGES, t, phase);
        __builtin_amdgcn_sched_barrier(0);
        if constexpr(decltype(refill)::value)
        {
            prefetch((t + T::INITIAL) % T::STAGES, t + T::INITIAL);
            __builtin_amdgcn_sched_barrier(0);
        }
        compute(f);
    };

    // ===== Prologue =====
    // Fill the register queue and initial LDS stages.
    static_for<T::B_DIRECT ? T::B_AHEAD : 0>([&](auto ti) {
        prefetch_b(decltype(ti)::value, number<decltype(ti)::value % T::B_SETS>{});
    });
    if constexpr(!T::PER_STAGE)
        prefetch_scales(0, 0);
    static_for<T::INITIAL>([&](auto si) { prefetch(decltype(si)::value, decltype(si)::value); });

    // ===== Main loop =====
    constexpr int steady = T::PANELS - T::INITIAL;
#pragma clang loop unroll(disable)
    for(int i = 0; i < steady / T::UNROLL; ++i)
    {
        static_for<T::UNROLL>([&](auto pi) {
            step(i * T::UNROLL + decltype(pi)::value, pi, number<T::INITIAL - 1>{}, 1_I, 1_I);
        });
    }
    static_for<steady % T::UNROLL>([&](auto ii) {
        constexpr int t = steady - steady % T::UNROLL + decltype(ii)::value;
        step(t, number<t % T::B_SETS>{}, number<T::INITIAL - 1>{}, 1_I, 1_I);
    });

    // ===== Epilogue =====
    // Drain the initial LDS queue with decreasing pending counts.
    static_for<T::INITIAL>([&](auto ji) {
        constexpr int j = decltype(ji)::value;
        constexpr int t = steady + j;
        step(t,
             number<t % T::B_SETS>{},
             number<T::INITIAL - 1 - j>{},
             0_I,
             number<(t + T::B_AHEAD < T::PANELS)>{});
    });

    // ===== Write back =====
    // The direct and fused split-K paths use the same C layouts.
    auto store_output = [&]() {
        auto g_c = make_gmem(final_output + static_cast<int64_t>(row) * stride_c + col,
                             ((valid_rows - 1) * stride_c + T::B_N) * sizeof(D_C));
        if constexpr(T::OUTPUT < 2)
        {
            static_for<T::E_M * T::E_N>([&](auto fi) {
                constexpr int f  = decltype(fi)::value;
                constexpr int mi = f / T::E_N;
                constexpr int ni = f % T::E_N;
                auto values      = cast<D_C>(c[f]);
                const auto u_rc  = layout::make_layout_rc<T::NORMAL, T::B_M, T::B_N>(
                    mfrag(number<mi>{}) * 16, nfrag(number<ni>{}) * 16, lane_id, stride_c);
                store<T::NORMAL ? 1 : 4>(g_c, values, u_rc, 0, 0_I);
            });
        }
        else
        {
            constexpr int pitch = T::B_N + 8;
            auto staged         = make_smem(reinterpret_cast<D_C*>(smem));
            sync_threads();
            static_for<T::E_M * T::E_N>([&](auto fi) {
                constexpr int f  = decltype(fi)::value;
                constexpr int mi = f / T::E_N;
                constexpr int ni = f % T::E_N;
                auto values      = cast<D_C>(c[f]);
                const auto u_rc  = layout::make_layout_rc<T::NORMAL, T::B_M, T::B_N>(
                    mfrag(number<mi>{}) * 16, nfrag(number<ni>{}) * 16, lane_id, pitch);
                store<T::NORMAL ? 1 : 4>(staged, values, u_rc);
            });
            s_waitcnt_lgkmcnt(0_I);
            sync_threads();
            constexpr int chunks = T::B_N / 8;
            static_for<(T::B_M * chunks + T::BLOCK_SIZE - 1) / T::BLOCK_SIZE>([&](auto ii) {
                const int index = decltype(ii)::value * T::BLOCK_SIZE + tid;
                const int r     = index / chunks;
                const int col   = (index % chunks) * 8;
                if(index < T::B_M * chunks)
                {
                    const auto u_sc =
                        layout::make_layout_matrix_vector<8, T::B_M, T::B_N>(r, col, pitch);
                    const auto u_gc =
                        layout::make_layout_matrix_vector<8, T::B_M, T::B_N>(r, col, stride_c);
                    store<8>(
                        g_c, load<8>(staged, u_sc), u_gc, 0, number<(T::OUTPUT == 2 ? 2 : 0)>{});
                }
            });
        }
    };
    if constexpr(T::SPLIT_K == 1)
    {
        store_output();
    }
    else
    {
        // Continuous fragment slots keep both the producer and final gather
        // coalesced, including partial M tiles. Only final C is bounds-masked.
        constexpr int tile_elements = T::B_M * T::B_N;
        auto g_partials             = make_gmem(reinterpret_cast<D_ACC*>(kargs.ptr_c) +
                                        static_cast<int64_t>(tile) * T::SPLIT_K * tile_elements,
                                    T::SPLIT_K * tile_elements * sizeof(D_ACC));
        static_for<T::E_M * T::E_N>([&](auto fi) {
            constexpr int f = decltype(fi)::value;
            store<4>(g_partials,
                     c[f],
                     layout::make_layout_partial<T::BLOCK_SIZE, tile_elements, T::SPLIT_K>(
                         f, thread_id_x(), split),
                     0,
                     0_I);
        });
        // Local-L2 mode follows the gfx950 XCD-grouped publication protocol:
        // every wave drains its stores, all waves arrive, then the counter updates.
        s_waitcnt_vmcnt(0_I);
        asm volatile("" ::: "memory");
        sync_threads();
        __shared__ unsigned arrived;
        if(thread_id_x() == 0)
        {
            arrived = __hip_atomic_fetch_add(counters + tile, 1u, __ATOMIC_ACQ_REL, 3);
        }
        s_waitcnt_lgkmcnt(0_I);
        sync_threads();
        if(arrived == T::SPLIT_K - 1)
        {
            static_for<T::E_M * T::E_N>([&](auto fi) {
                constexpr int f = decltype(fi)::value;
                typename BaseMMA::vtype_c sum{};
                static_for<T::SPLIT_K>([&](auto si) {
                    constexpr int sk = decltype(si)::value;
                    sum += load<4>(
                        g_partials,
                        layout::make_layout_partial<T::BLOCK_SIZE, tile_elements, T::SPLIT_K>(
                            f, thread_id_x(), sk),
                        0,
                        number<17>{});
                });
                c[f] = sum;
            });
            store_output();
            if(thread_id_x() == 0)
            {
                __hip_atomic_store(counters + tile, 0u, __ATOMIC_RELAXED, 3);
            }
        }
    }
}
template <class Traits>
__device__ __forceinline__ void opus_gemm_blockscale_mxscale_bpreshuffle_composable_direct(
    opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 kargs)
{
    static_assert(Traits::SPLIT_K == 1);
    opus_gemm_blockscale_mxscale_bpreshuffle_composable<Traits>(
        kargs, reinterpret_cast<opus::bf16_t*>(kargs.ptr_c), nullptr);
}
#endif

// The existing tile and fused split launchers share this pipeline body.
template <class Traits>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, Traits::WG_PER_CU) void
opus_gemm_composable_split_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 kargs,
                                  opus::bf16_t* g_c,
                                  unsigned* counters)
{
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
    static_assert(Traits::SPLIT_K > 1);
    opus_gemm_blockscale_mxscale_bpreshuffle_composable<Traits>(kargs, g_c, counters);
#endif
}
template <class Traits>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, Traits::WG_PER_CU) void
opus_gemm_composable_direct_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 kargs)
{
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
    opus_gemm_blockscale_mxscale_bpreshuffle_composable_direct<Traits>(kargs);
#endif
}
