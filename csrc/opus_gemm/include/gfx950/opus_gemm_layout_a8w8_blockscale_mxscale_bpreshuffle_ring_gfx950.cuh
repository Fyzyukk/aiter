// SPDX-License-Identifier: Apache-2.0
#pragma once

#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
#include "../opus_gemm_utils.cuh"

namespace opus_gemm_blockscale_mxscale_bpreshuffle_ring_detail {
using opus::operator""_I;

// The ring pipelines count DMA groups, rather than individual VMEM operations.
// Keep the grouped instruction behind the same view/layout API as opus::gmem;
// the ordinary gmem transport is not tracked by asyncmark on gfx950.
template <class D>
struct async_gmem : opus::gmem<D>
{
    using Base = opus::gmem<D>;
    using Base::Base;

    template <opus::index_t Vec, opus::index_t Immediate = 0, opus::index_t Cache = 0>
    __device__ __forceinline__ void _async_load(OPUS_LDS_ADDR void* destination,
                                                int offset,
                                                int scalar_offset       = 0,
                                                opus::number<Immediate> = {},
                                                opus::number<Cache>     = {})
    {
        using Value = typename Base::template vector_type<Vec>;
        __builtin_amdgcn_raw_ptr_buffer_load_async_lds(this->cached_rsrc,
                                                       destination,
                                                       sizeof(Value),
                                                       offset * sizeof(D),
                                                       scalar_offset * sizeof(D),
                                                       Immediate,
                                                       Cache);
    }

    template <opus::index_t Vec, opus::index_t Immediate = 0, opus::index_t Cache = 0>
    __device__ __forceinline__ void async_load(void* destination,
                                               int offset,
                                               int scalar_offset       = 0,
                                               opus::number<Immediate> = {},
                                               opus::number<Cache>     = {})
    {
        auto* target =
            reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(destination));
        _async_load<Vec>(
            target, offset, scalar_offset, opus::number<Immediate>{}, opus::number<Cache>{});
    }

    template <opus::index_t Vec,
              class GlobalLayout,
              class SharedLayout,
              opus::index_t Immediate = 0,
              opus::index_t Cache     = 0,
              std::enable_if_t<opus::is_layout_v<GlobalLayout> && opus::is_layout_v<SharedLayout>,
                               bool>  = true>
    __device__ __forceinline__ void async_load(OPUS_LDS_ADDR void* destination,
                                               const GlobalLayout& global_layout,
                                               const SharedLayout& shared_layout,
                                               int scalar_offset       = 0,
                                               opus::number<Immediate> = {},
                                               opus::number<Cache>     = {})
    {
        using Load                = opus::layout_load_traits<GlobalLayout, Vec>;
        const auto global_offsets = opus::layout_to_offsets<Vec>(global_layout);
        const auto shared_offsets = opus::layout_to_offsets<Vec>(shared_layout);
        // Preserve the OPUS shared view's address space through layout offsets.
        // A generic-pointer round trip adds a null-address select at each DMA.
        auto* target = reinterpret_cast<OPUS_LDS_ADDR D*>(destination);
        opus::static_for<Load::r_elem.value>([&](auto i) {
            _async_load<Vec>(target + shared_offsets[i],
                             global_offsets[i],
                             scalar_offset,
                             opus::number<Immediate>{},
                             opus::number<Cache>{});
        });
    }
};

template <class D>
__device__ __forceinline__ auto make_async_gmem(const D* ptr, unsigned int bytes)
{
    return async_gmem<D>(ptr, bytes);
}

__device__ __forceinline__ void commit_async_loads() { __builtin_amdgcn_asyncmark(); }

template <int Pending>
__device__ __forceinline__ void wait_async_loads(opus::number<Pending>)
{
    __builtin_amdgcn_wait_asyncmark(Pending);
}

// Some scale-address policies require a VGPR offset even when all active rows
// happen to share a scalar address. Preserve that requirement at the transport.
template <class Layout>
__device__ __forceinline__ auto vectorize_layout_offsets(Layout layout)
{
#if defined(__AMDGCN__)
    for(int i = 0; i < Layout::num_issues; ++i)
    {
        asm volatile("" : "+v"(layout.offsets[i]));
    }
#endif
    return layout;
}

// Assemble the two MFMA chunks directly, as in the pinned-AGPR pipeline.
// Static chunk ownership avoids a register-array index in the LDS load path.
template <class Fragment, class Memory, class Layout>
__device__ __forceinline__ Fragment load_operand_fragment(Memory& memory, const Layout& layout)
{
    using Load = opus::layout_load_traits<Layout, 16>;
    static_assert(Load::r_elem.value == 2);
    const auto offsets = opus::layout_to_offsets<16>(layout);
    using Chunk        = decltype(opus::load<16>(memory, offsets[0]));
    opus::array<Chunk, 2> values;
    opus::static_for<2>([&](auto i) { values[i] = opus::load<16>(memory, offsets[i]); });
    Fragment result;
    opus::set_slice(result, values[0], 0_I, 16_I);
    opus::set_slice(result, values[1], 16_I, 32_I);
    return result;
}

// gfx950 DMA supplies the lane offset implicitly. The LDS layout therefore
// describes each wave's base, with one dword per lane for byte-scale copies.
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
    return vectorize_layout_offsets(layout);
}

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

__device__ inline constexpr auto
make_layout_gsfb_fragment(int group_n, int column, int groups, int stride_sfb)
{
    const auto gsfb_block_shape   = opus::make_tuple(groups, stride_sfb, 1_I);
    constexpr auto gsfb_block_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}), opus::make_tuple(opus::p_dim{}, opus::y_dim{}));
    return opus::make_layout<1>(
        gsfb_block_shape,
        opus::unfold_x_stride(gsfb_block_dim, gsfb_block_shape, opus::tuple{stride_sfb, 1_I}),
        opus::unfold_p_coord(gsfb_block_dim, opus::tuple{group_n, column}));
}

template <bool Contiguous>
__device__ inline constexpr int repeat_index(int wave_id, int repeat, int repeats, int waves)
{
    if constexpr(Contiguous)
        return wave_id * repeats + repeat;
    else
        return repeat * waves + wave_id;
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
        return vectorize_layout_offsets(layout);
    else
        return layout;
}

template <int BlockM>
__device__ inline constexpr auto
make_layout_ga_fragment(int row, int lane_id, int k_slice, int stride_a)
{
    const auto ga_block_shape =
        opus::make_tuple(opus::number<BlockM / 16>{}, 16_I, stride_a / 128, 2_I, 4_I, 16_I);
    constexpr auto ga_block_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));
    return opus::make_layout<16>(
        ga_block_shape,
        opus::unfold_x_stride(ga_block_dim, ga_block_shape, opus::tuple{stride_a, 1_I}),
        opus::unfold_p_coord(ga_block_dim, opus::tuple{row / 16, row % 16, k_slice, lane_id / 16}));
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

} // namespace opus_gemm_blockscale_mxscale_bpreshuffle_ring_detail

namespace opus {
template <class D>
struct is_gmem<opus_gemm_blockscale_mxscale_bpreshuffle_ring_detail::async_gmem<D>> : true_type
{
};
} // namespace opus
#endif
