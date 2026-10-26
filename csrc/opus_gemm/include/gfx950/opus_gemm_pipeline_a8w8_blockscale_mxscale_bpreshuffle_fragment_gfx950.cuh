// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "../opus_gemm_utils.cuh"
#include "opus_gemm_kargs_a8w8_mxscale_bpreshuffle_gfx950.cuh"
#include "opus_gemm_layout_a8w8_blockscale_mxscale_bpreshuffle_ring_gfx950.cuh"
#include "opus_gemm_traits_a8w8_blockscale_mxscale_bpreshuffle_fragment_gfx950.cuh"

#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
namespace opus_gemm_bpreshuffle_fragment_layout {
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

// Packed scale A layouts.
template <class T>
__device__ inline constexpr auto make_layout_ssfa_packed(int row, int k_slice, int columns)
{
    const auto ssfa_block_shape =
        opus::make_tuple(columns, number<T::T_M>{}, number<T::M_WORDS>{}, 4_I, 4_I, 4_I, 1_I);
    constexpr auto ssfa_block_dim =
        opus::make_tuple(opus::make_tuple(p_dim{}),
                         opus::make_tuple(p_dim{}, p_dim{}, p_dim{}, y_dim{}),
                         opus::make_tuple(p_dim{}, y_dim{}));
    const int repeat = (row / 16) % T::E_M;
    return opus::make_layout<1>(
               ssfa_block_shape,
               opus::unfold_x_stride(
                   ssfa_block_dim, ssfa_block_shape, tuple{number<T::SCALE_M_BYTES>{}, 4_I, 1_I}),
               opus::unfold_p_coord(
                   ssfa_block_dim,
                   tuple{k_slice, row / (T::E_M * 16), repeat / 4, (row % 16) / 4, repeat % 4})) +
           number<T::SCALE_A_OFFSET>{};
}

template <class T>
__device__ inline constexpr auto
make_layout_rsfa_packed(int wave_id_m, int word, int lane_id, int k_slice, int columns)
{
    const auto rsfa_block_shape =
        opus::make_tuple(columns, number<T::T_M>{}, number<T::M_WORDS>{}, 16_I, 1_I);
    constexpr auto rsfa_block_dim = opus::make_tuple(
        opus::make_tuple(p_dim{}), opus::make_tuple(p_dim{}, p_dim{}, p_dim{}, y_dim{}));
    return opus::make_layout<1>(
               rsfa_block_shape,
               opus::unfold_x_stride(
                   rsfa_block_dim, rsfa_block_shape, tuple{number<T::SCALE_M_BYTES / 4>{}, 1_I}),
               opus::unfold_p_coord(rsfa_block_dim,
                                    tuple{k_slice, wave_id_m, word, lane_id % 16})) +
           number<T::SCALE_A_OFFSET / 4>{};
}

// Scale B layouts.
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

// Packed scale B layouts.
template <class T>
__device__ inline constexpr auto make_layout_rsfb_packed(int k_slice, int columns, int base)
{
    const auto rsfb_block_shape = opus::make_tuple(number<(T::N_SCALES + 3) / 4>{}, columns, 1_I);
    constexpr auto rsfb_block_dim =
        opus::make_tuple(opus::make_tuple(p_dim{}), opus::make_tuple(p_dim{}, y_dim{}));
    return opus::make_layout<1>(
               rsfb_block_shape,
               opus::unfold_x_stride(rsfb_block_dim, rsfb_block_shape, tuple{columns, 1_I}),
               opus::unfold_p_coord(rsfb_block_dim, tuple{0_I, k_slice})) +
           base;
}

// C layouts.
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
} // namespace opus_gemm_bpreshuffle_fragment_layout

namespace opus_gemm_blockscale_mxscale_bpreshuffle_fragment_detail {
using namespace opus;
using namespace opus_gemm_blockscale_mxscale_bpreshuffle_ring_detail;
namespace layout = opus_gemm_bpreshuffle_fragment_layout;

// The memory layouts describe the two 16-byte halves of a scaled-MFMA
// operand. No intermediate LDS representation or workgroup barrier is needed.
template <class T>
struct fragment_mma
{
    using MMA =
        decltype(make_mfma<fp8_t, fp8_t, fp32_t>(seq<16, 16, 128>{}, mfma_adaptor_swap_ab{}));
    struct OperandFragments
    {
        array<typename MMA::vtype_a, T::E_M> a;
        array<typename MMA::vtype_b, T::E_N> b;
        array<unsigned, T::E_M> sa;
        array<unsigned, T::N_SCALES> sb;
    };
    using CTile = array<typename MMA::vtype_c, T::E_M * T::E_N>;

    template <int Count, class Scales>
    __device__ static auto scale_words(const Scales& scales)
    {
        constexpr int bytes = T::PACK_SCALES ? 4 : 1;
        array<unsigned, (Count + bytes - 1) / bytes> result{};
        static_for<Count>([&](auto i) {
            constexpr int j = decltype(i)::value;
            result[j / bytes] |= scales[j] << ((j % bytes) * 8);
        });
        return result;
    }

    template <bool LoadScales = true, class GA, class GB, class GSA, class GSB>
    __device__ static OperandFragments load_operands(GA& g_a,
                                                     GB& g_b,
                                                     GSA& g_sfa,
                                                     GSB& g_sfb,
                                                     int kblock,
                                                     int stride_a,
                                                     int stride_b,
                                                     int stride_sa,
                                                     int stride_sb,
                                                     int row,
                                                     int col,
                                                     int shape_m,
                                                     int shape_n,
                                                     int lane_id,
                                                     int wave_id_m,
                                                     int wave_id_n)
    {
        OperandFragments p;
        if constexpr(!LoadScales)
        {
            p.sa = {};
            p.sb = {};
        }
        const int mr = wave_id_m * T::E_M * 16 + lane_id % 16;
        static_for<T::E_M>([&](auto i) {
            constexpr int m = decltype(i)::value;
            const int r     = mr + m * 16;
            const auto u_ga = layout::make_layout_ga_fragment<T::B_M>(r, lane_id, kblock, stride_a);
            p.a[m]          = load<16>(g_a, u_ga, 0, number<T::CACHE_A>{});
            // A is zero outside M. Bound scale addressing separately so a tail
            // row cannot accidentally consume the next K block's exponent.
            if constexpr(LoadScales)
            {
                const int sr      = row + r < shape_m ? row + r : row;
                const auto u_gsfa = layout::make_layout_gsfa_fragment<1>(
                    sr, kblock, shape_m, stride_a / 128, stride_sa);
                p.sa[m] = load<1>(g_sfa, u_gsfa)[0];
            }
        });
        static_for<T::E_N>([&](auto i) {
            constexpr int n = decltype(i)::value;
            const int ng    = n * T::T_N + wave_id_n;
            const auto u_gb = layout::make_layout_rb<T::B_N>(ng, lane_id, kblock, stride_b);
            p.b[n]          = load<16>(g_b, u_gb, 0, number<T::CACHE_B>{});
        });
        if constexpr(LoadScales)
        {
            static_for<T::N_SCALES>([&](auto i) {
                constexpr int s   = decltype(i)::value;
                const auto u_gsfb = layout::make_layout_gsfb_fragment(
                    col / 128 + s, kblock, shape_n / 128, stride_sb);
                p.sb[s] = load<1>(g_sfb, u_gsfb)[0];
            });
        }
        return p;
    }

    __device__ static void compute_mfma(const OperandFragments& p, CTile& c)
    {
        const auto sa = [&] {
            if constexpr(T::SCALES_PREPACKED)
                return p.sa;
            else
                return scale_words<T::E_M>(p.sa);
        }();
        const auto sb = [&] {
            if constexpr(T::SCALES_PREPACKED)
                return p.sb;
            else
                return scale_words<T::N_SCALES>(p.sb);
        }();
        constexpr int bytes = T::PACK_SCALES ? 4 : 1;
        static_for<T::E_M>([&](auto mi) {
            constexpr int m = decltype(mi)::value;
            static_for<T::E_N>([&](auto ni) {
                constexpr int n   = decltype(ni)::value;
                constexpr int ns  = n / T::N_PER_SCALE;
                c[m * T::E_N + n] = MMA{}(p.a[m],
                                          p.b[n],
                                          c[m * T::E_N + n],
                                          int(sa[m / bytes]),
                                          int(sb[ns / bytes]),
                                          number<m % bytes>{},
                                          number<ns % bytes>{});
            });
        });
    }
};

// A shared-memory transport for the same MFMA operands. Consume the complete
// K panel before releasing its slot while the next panel remains in flight.
template <class T>
__device__ void shared_transport(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 kargs)
{
    using Engine      = fragment_mma<T>;
    const int row     = block_id_y() * T::B_M;
    const int col     = block_id_x() * T::B_N;
    const int tid     = thread_id_x();
    const int lane_id = tid % 64;
    __builtin_assume(static_cast<unsigned>(tid) < T::BLOCK_SIZE);
    const int wave_id   = __builtin_amdgcn_readfirstlane(tid / 64);
    const int wave_id_m = wave_id / T::T_N;
    const int wave_id_n = wave_id % T::T_N;
    const int rows      = kargs.m - row < T::B_M ? kargs.m - row : T::B_M;
    const int sc        = kargs.k / 128;
    auto g_a =
        make_async_gmem(reinterpret_cast<const fp8_t*>(kargs.ptr_a) + int64_t(row) * kargs.stride_a,
                        rows * kargs.stride_a);
    auto g_b =
        make_async_gmem(reinterpret_cast<const fp8_t*>(kargs.ptr_b) + int64_t(col) * kargs.stride_b,
                        T::B_N * kargs.stride_b);
    auto g_sfa =
        make_async_gmem(reinterpret_cast<const unsigned char*>(kargs.ptr_sfa), kargs.m * sc);
    auto g_sfb = make_async_gmem(reinterpret_cast<const unsigned char*>(kargs.ptr_sfb),
                                 (kargs.n / 128) * sc);
    extern __shared__ __attribute__((aligned(16))) unsigned char smem[];
    auto s_ab           = make_smem(reinterpret_cast<fp8_t*>(smem));
    auto s_sf           = make_smem(smem);
    const int sb_offset = T::SCALE_A_OFFSET + T::scale_a_bytes(kargs.k);

    // ===== Prologue =====
    // Publish scales before filling the first two data slots.
    if constexpr(T::SCALES_PREPACKED)
    {
        // Transpose the small scale tile once into words owned by an MFMA
        // lane. The inner K loop then reads one dword for four M repeats.
        const int copies = (T::B_M * sc + T::BLOCK_SIZE * 4 - 1) / (T::BLOCK_SIZE * 4);
        for(int i = 0; i < copies; ++i)
        {
            const int index = (i * T::BLOCK_SIZE + tid) * 4;
            const int r     = index % T::B_M;
            const int ks    = index / T::B_M;
            if(ks < sc)
            {
                const bool valid  = row + r < kargs.m;
                const auto u_gsfa = layout::make_layout_gsfa_fragment<4>(
                    valid ? row + r : kargs.m, valid ? ks : sc - 1, kargs.m, sc, kargs.m);
                const auto u_ssfa = layout::make_layout_ssfa_packed<T>(r, ks, sc);
                store<1>(s_sf, load<4>(g_sfa, u_gsfa), u_ssfa);
            }
        }
        for(int ks = tid; ks < sc; ks += T::BLOCK_SIZE)
        {
            unsigned word = 0;
            static_for<T::N_SCALES>([&](auto ni) {
                constexpr int ns  = decltype(ni)::value;
                const auto u_gsfb = layout::make_layout_gsfb_fragment(
                    col / 128 + ns, ks, kargs.n / 128, kargs.stride_sfb);
                word |= unsigned(load<1>(g_sfb, u_gsfb)[0]) << (ns * 8);
            });
            auto words = make_smem(reinterpret_cast<unsigned*>(smem));
            store<1>(words,
                     make_repeated_vector(word, 1_I),
                     layout::make_layout_rsfb_packed<T>(ks, sc, sb_offset / 4));
        }
        s_waitcnt_lgkmcnt(0_I);
    }
    else
    {
        // A's column-major exponents are copied as compact dwords. Host eligibility
        // requires M % 4 == 0; invalid whole dwords are ignored by the A/C bounds.
        const int a_copies = T::scale_a_bytes(kargs.k) / T::SCALE_A_DMA;
        for(int i = 0; i < a_copies; ++i)
        {
            const int index   = (i * T::BLOCK_SIZE + tid) * 4;
            const int r       = index % T::B_M;
            const int ks      = index / T::B_M;
            const bool valid  = ks < sc && row + r < kargs.m;
            const auto u_gsfa = layout::make_layout_gsfa_fragment<4>(
                valid ? row + r : kargs.m, valid ? ks : sc - 1, kargs.m, sc, kargs.m);
            const auto u_ssfa =
                layout::make_layout_dma<4, T::BLOCK_SIZE>(i, wave_id, T::SCALE_A_OFFSET, a_copies);
            async_load<4>(g_sfa, s_sf.ptr, u_gsfa, u_ssfa);
        }
        const int b_copies = T::scale_b_bytes(kargs.k) / (T::BLOCK_SIZE * 4);
        for(int i = 0; i < b_copies; ++i)
        {
            const int index  = i * T::BLOCK_SIZE + tid;
            const int ng     = index / sc;
            const int ks     = index % sc;
            const bool valid = ng < (T::B_N + 127) / 128;
            const auto u_gsfb =
                layout::make_layout_gsfb_fragment(valid ? col / 128 + ng : kargs.n / 128,
                                                  valid ? ks : 0,
                                                  kargs.n / 128,
                                                  kargs.stride_sfb);
            const auto u_ssfb =
                layout::make_layout_dma<1, T::BLOCK_SIZE>(i, wave_id, sb_offset, b_copies);
            async_load<1>(g_sfb, s_sf.ptr, u_gsfb, u_ssfb);
        }
    }
    using BFragment = typename Engine::MMA::vtype_b;
    array<array<array<BFragment, T::E_N>, T::STAGE_K / 128>, 2> b_queue;
    auto prefetch = [&](auto slot_c, int panel) {
        constexpr int slot = decltype(slot_c)::value;
        // Keep the uniform K-panel displacement in the scalar load offset.
        // The per-lane layout then needs no separate VGPR induction per copy.
        static_for<T::A_BYTES / T::DMA>([&](auto i) {
            const auto u_ga = layout::make_layout_ga<T::B_M, T::STAGE_K, T::BLOCK_SIZE>(
                i, lane_id, wave_id, kargs.stride_a);
            const auto u_sa = layout::make_layout_dma<16, T::BLOCK_SIZE>(
                i, wave_id, slot * T::PANEL_BYTES, T::A_BYTES / T::DMA);
            async_load<16>(g_a, s_ab.ptr, u_ga, u_sa, panel * T::STAGE_K);
        });
        static_for<T::B_BYTES / T::DMA>([&](auto i) {
            const auto u_gb = layout::make_layout_gb<T::B_N, T::STAGE_K, T::BLOCK_SIZE>(
                i, lane_id, wave_id, kargs.stride_b);
            const auto u_sb = layout::make_layout_dma<16, T::BLOCK_SIZE>(
                i, wave_id, slot * T::PANEL_BYTES + T::A_BYTES, T::B_BYTES / T::DMA);
            async_load<16>(g_b, s_ab.ptr, u_gb, u_sb, panel * T::STAGE_K * 16);
        });
        if constexpr(T::REGISTER_B)
        {
            static_for<T::STAGE_K / 128>([&](auto ki) {
                constexpr int ks = decltype(ki)::value;
                static_for<T::E_N>([&](auto ni) {
                    constexpr int n = decltype(ni)::value;
                    const auto u_gb =
                        layout::make_layout_rb<T::B_N>(n * T::T_N + wave_id_n,
                                                       lane_id,
                                                       panel * (T::STAGE_K / 128) + ks,
                                                       kargs.stride_b);
                    b_queue[slot][ks][n] = load<16>(g_b, u_gb);
                });
            });
        }
        commit_async_loads();
    };
    auto read = [&](auto slot_c, int panel, auto ki) {
        constexpr int slot = decltype(slot_c)::value;
        constexpr int ks   = decltype(ki)::value;
        typename Engine::OperandFragments p;
        const int global_ks = panel * (T::STAGE_K / 128) + ks;
        static_for<T::E_M>([&](auto mi) {
            constexpr int m = decltype(mi)::value;
            const int r     = (wave_id_m * T::E_M + m) * 16 + lane_id % 16;
            const auto u_ra =
                layout::make_layout_ra<T::B_M, T::STAGE_K>(r, lane_id, ks, slot * T::PANEL_BYTES);
            p.a[m] = load_operand_fragment<typename Engine::MMA::vtype_a>(s_ab, u_ra);
            if constexpr(T::SCALES_PREPACKED)
            {
                if constexpr(m % 4 == 0)
                {
                    auto words  = make_smem(reinterpret_cast<unsigned*>(smem));
                    p.sa[m / 4] = load<1>(words,
                                          layout::make_layout_rsfa_packed<T>(
                                              wave_id_m, m / 4, lane_id, global_ks, sc))[0];
                }
            }
            else
            {
                p.sa[m] = load<1>(
                    s_sf,
                    layout::make_layout_rsfa<4>(r, global_ks, T::B_M, sc, T::SCALE_A_OFFSET))[0];
            }
        });
        static_for<T::E_N>([&](auto ni) {
            constexpr int n = decltype(ni)::value;
            if constexpr(T::REGISTER_B)
                p.b[n] = b_queue[slot][ks][n];
            else
            {
                const auto u_rb =
                    layout::make_layout_rb<T::B_N>(n * T::T_N + wave_id_n,
                                                   lane_id,
                                                   ks,
                                                   T::STAGE_K,
                                                   slot * T::PANEL_BYTES + T::A_BYTES);
                p.b[n] = load_operand_fragment<typename Engine::MMA::vtype_b>(s_ab, u_rb);
            }
        });
        static_for<T::N_SCALES>([&](auto ni) {
            constexpr int ns = decltype(ni)::value;
            if constexpr(T::SCALES_PREPACKED)
            {
                if constexpr(ns == 0)
                {
                    auto words = make_smem(reinterpret_cast<unsigned*>(smem));
                    p.sb[0]    = load<1>(
                        words, layout::make_layout_rsfb_packed<T>(global_ks, sc, sb_offset / 4))[0];
                }
            }
            else
            {
                p.sb[ns] = load<1>(
                    s_sf, layout::make_layout_rsfb<T::N_SCALES>(ns, global_ks, sc, sb_offset))[0];
            }
        });
        return p;
    };
    typename Engine::CTile c{};
    const int panels = (kargs.k + T::STAGE_K - 1) / T::STAGE_K;
    prefetch(0_I, 0);
    if(panels > 1)
        prefetch(1_I, 1);

        // ===== Main loop =====
        // The dynamic panel loop keeps its original per-slot waits and final drain.
#pragma clang loop unroll(disable)
    for(int base = 0; base < panels; base += 2)
    {
        static_for<2>([&](auto slot) {
            const int panel = base + decltype(slot)::value;
            if(panel < panels)
            {
                if(panel + 1 < panels)
                    wait_async_loads(1_I);
                else
                    wait_async_loads(0_I);
                sync_threads();
                static_for<T::STAGE_K / 128>([&](auto ki) {
                    if(panel * (T::STAGE_K / 128) + decltype(ki)::value < sc)
                    {
                        auto p = read(slot, panel, ki);
                        Engine::compute_mfma(p, c);
                    }
                });
                // All consumers release this slot before any wave refills it.
                sync_threads();
                if(panel + 2 < panels)
                    prefetch(slot, panel + 2);
            }
        });
    }

    // ===== Epilogue =====
    // The panel queue is drained; prepare the bounded C view.
    auto g_c =
        make_gmem(reinterpret_cast<bf16_t*>(kargs.ptr_c) + int64_t(row) * kargs.stride_c + col,
                  ((rows - 1) * kargs.stride_c + T::B_N) * sizeof(bf16_t));

    // ===== Write back =====
    static_for<T::E_M>([&](auto mi) {
        constexpr int m = decltype(mi)::value;
        static_for<T::E_N>([&](auto ni) {
            constexpr int n = decltype(ni)::value;
            const auto u_rc =
                layout::make_layout_rc<false, T::B_M, T::B_N>((wave_id_m * T::E_M + m) * 16,
                                                              (n * T::T_N + wave_id_n) * 16,
                                                              lane_id,
                                                              kargs.stride_c);
            store<4>(g_c, cast<bf16_t>(c[m * T::E_N + n]), u_rc);
        });
    });
}
template <class T>
__device__ auto
scheduled_compute(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 kargs, int row, int col)
{
    using Engine      = fragment_mma<T>;
    const int tid     = thread_id_x();
    const int lane_id = tid % 64;
    __builtin_assume(static_cast<unsigned>(tid) < T::BLOCK_SIZE);
    const int wave_id   = __builtin_amdgcn_readfirstlane(tid / 64);
    const int wave_id_m = wave_id / T::T_N;
    const int wave_id_n = wave_id % T::T_N;
    const int rows      = kargs.m - row < T::B_M ? kargs.m - row : T::B_M;
    auto g_a            = make_async_gmem(
        reinterpret_cast<const fp8_t*>(kargs.ptr_a) + int64_t(row) * T::FIXED_K, rows * T::FIXED_K);
    auto g_b =
        make_async_gmem(reinterpret_cast<const fp8_t*>(kargs.ptr_b) + int64_t(col) * T::FIXED_K,
                        T::B_N * T::FIXED_K);
    auto g_sfa = make_async_gmem(reinterpret_cast<const unsigned char*>(kargs.ptr_sfa),
                                 kargs.m * (T::FIXED_K / 128));
    auto g_sfb = make_async_gmem(reinterpret_cast<const unsigned char*>(kargs.ptr_sfb),
                                 (kargs.n / 128) * (T::FIXED_K / 128));
    array<array<unsigned, T::E_M>, T::PANELS> sa;
    array<array<unsigned, T::N_SCALES>, T::PANELS> sb;
    auto load_operands = [&](int panel) {
        return Engine::template load_operands<!T::PRELOAD_SCALES>(g_a,
                                                                  g_b,
                                                                  g_sfa,
                                                                  g_sfb,
                                                                  panel,
                                                                  T::FIXED_K,
                                                                  T::FIXED_K,
                                                                  kargs.stride_sfa,
                                                                  T::FIXED_K / 128,
                                                                  row,
                                                                  col,
                                                                  kargs.m,
                                                                  kargs.n,
                                                                  lane_id,
                                                                  wave_id_m,
                                                                  wave_id_n);
    };
    typename Engine::CTile c{};
    array<typename Engine::OperandFragments, T::DEPTH> queue;

    auto step = [&](auto i) {
        constexpr int panel = decltype(i)::value;
        constexpr int slot  = panel % T::DEPTH;
        if constexpr(T::PRELOAD_SCALES)
        {
            queue[slot].sa = sa[panel];
            queue[slot].sb = sb[panel];
        }
        Engine::compute_mfma(queue[slot], c);
        __builtin_amdgcn_sched_barrier(0);
        if constexpr(panel + T::DEPTH < T::PANELS)
            queue[slot] = load_operands(panel + T::DEPTH);
        __builtin_amdgcn_sched_barrier(0);
    };

    // ===== Prologue =====
    // Preload the requested scale policy, then fill the operand queue.
    if constexpr(T::PRELOAD_SCALES)
    {
        static_for<T::PANELS>([&](auto ki) {
            constexpr int panel = decltype(ki)::value;
            const int kblock    = panel;
            static_for<T::E_M>([&](auto mi) {
                constexpr int m   = decltype(mi)::value;
                const int r       = row + (wave_id_m * T::E_M + m) * 16 + lane_id % 16;
                const auto u_gsfa = layout::make_layout_gsfa_fragment<1>(
                    r < kargs.m ? r : row, kblock, kargs.m, T::PANELS, kargs.stride_sfa);
                sa[panel][m] = load<1>(g_sfa, u_gsfa)[0];
            });
            static_for<T::N_SCALES>([&](auto si) {
                const auto u_gsfb = layout::make_layout_gsfb_fragment(
                    col / 128 + decltype(si)::value, kblock, kargs.n / 128, T::PANELS);
                sb[panel][si] = load<1>(g_sfb, u_gsfb)[0];
            });
        });
        __builtin_amdgcn_sched_barrier(0);
    }
    static_for<(T::PANELS < T::DEPTH ? T::PANELS : T::DEPTH)>(
        [&](auto i) { queue[i] = load_operands(decltype(i)::value); });
    __builtin_amdgcn_sched_barrier(0);

    // ===== Main loop =====
    constexpr int initial = T::PANELS < T::DEPTH ? T::PANELS : T::DEPTH;
    constexpr int steady  = T::PANELS - initial;
    static_for<steady>([&](auto i) { step(i); });

    // ===== Epilogue =====
    // Retire the remaining operand queue without another refill.
    static_for<initial>([&](auto i) { step(number<steady + decltype(i)::value>{}); });
    return c;
}

template <class T, class C>
__device__ void
scheduled_store(C& c, bf16_t* destination, int row, int col, int shape_m, int stride_c)
{

    // ===== Write back =====
    // Convert the completed scheduled fragments and store bounded C.
    const int tid       = thread_id_x();
    const int lane_id   = tid % 64;
    const int wave_id   = __builtin_amdgcn_readfirstlane(tid / 64);
    const int wave_id_m = wave_id / T::T_N;
    const int wave_id_n = wave_id % T::T_N;
    const int rows      = shape_m - row < T::B_M ? shape_m - row : T::B_M;
    auto g_c            = make_gmem(destination + int64_t(row) * stride_c + col,
                         ((rows - 1) * stride_c + T::B_N) * sizeof(bf16_t));
    static_for<T::E_M>([&](auto mi) {
        constexpr int m = decltype(mi)::value;
        static_for<T::E_N>([&](auto ni) {
            constexpr int n = decltype(ni)::value;
            const auto u_rc = layout::make_layout_rc<false, T::B_M, T::B_N>(
                (wave_id_m * T::E_M + m) * 16, (n * T::T_N + wave_id_n) * 16, lane_id, stride_c);
            store<4>(g_c, cast<bf16_t>(c[m * T::E_N + n]), u_rc, 0, number<2>{});
        });
    });
}
} // namespace opus_gemm_blockscale_mxscale_bpreshuffle_fragment_detail

template <class Traits>
__device__ __forceinline__ void opus_gemm_blockscale_mxscale_bpreshuffle_scheduled_fragment(
    opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 kargs)
{
    using namespace opus;
    using namespace opus_gemm_blockscale_mxscale_bpreshuffle_ring_detail;
    namespace layout = opus_gemm_bpreshuffle_fragment_layout;
    using T          = opus::remove_cvref_t<Traits>;
    const int row    = block_id_y() * T::B_M;
    const int col    = block_id_x() * T::B_N;
    auto c = opus_gemm_blockscale_mxscale_bpreshuffle_fragment_detail::scheduled_compute<T>(
        kargs, row, col);
    opus_gemm_blockscale_mxscale_bpreshuffle_fragment_detail::scheduled_store<T>(
        c, reinterpret_cast<bf16_t*>(kargs.ptr_c), row, col, kargs.m, kargs.stride_c);
}

template <class Traits>
__device__ __forceinline__ void opus_gemm_blockscale_mxscale_bpreshuffle_fragment(
    opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 kargs)
{
    using namespace opus;
    using namespace opus_gemm_blockscale_mxscale_bpreshuffle_ring_detail;
    namespace layout = opus_gemm_bpreshuffle_fragment_layout;
    using T          = opus::remove_cvref_t<Traits>;
    if constexpr(T::FIXED_K > 0)
    {
        kargs.k          = T::FIXED_K;
        kargs.stride_a   = T::FIXED_K;
        kargs.stride_b   = T::FIXED_K;
        kargs.stride_sfb = T::FIXED_K / 128;
    }
    if constexpr(T::STAGE_K > 0)
    {
        opus_gemm_blockscale_mxscale_bpreshuffle_fragment_detail::shared_transport<T>(kargs);
    }
    else
    {
        using Engine  = opus_gemm_blockscale_mxscale_bpreshuffle_fragment_detail::fragment_mma<T>;
        const int row = block_id_y() * T::B_M;
        const int col = block_id_x() * T::B_N;
        const int tid = thread_id_x();
        const int lane_id = tid % 64;
        __builtin_assume(static_cast<unsigned>(tid) < T::BLOCK_SIZE);
        const int wave_id   = __builtin_amdgcn_readfirstlane(tid / 64);
        const int wave_id_m = wave_id / T::T_N;
        const int wave_id_n = wave_id % T::T_N;
        const int rows      = kargs.m - row < T::B_M ? kargs.m - row : T::B_M;
        auto g_a            = make_async_gmem(reinterpret_cast<const fp8_t*>(kargs.ptr_a) +
                                       int64_t(row) * kargs.stride_a,
                                   rows * kargs.stride_a);
        auto g_b            = make_async_gmem(reinterpret_cast<const fp8_t*>(kargs.ptr_b) +
                                       int64_t(col) * kargs.stride_b,
                                   T::B_N * kargs.stride_b);
        auto g_sfa          = make_async_gmem(reinterpret_cast<const unsigned char*>(kargs.ptr_sfa),
                                     kargs.m * (kargs.k / 128));
        auto g_sfb          = make_async_gmem(reinterpret_cast<const unsigned char*>(kargs.ptr_sfb),
                                     (kargs.n / 128) * (kargs.k / 128));
        auto g_c =
            make_gmem(reinterpret_cast<bf16_t*>(kargs.ptr_c) + int64_t(row) * kargs.stride_c + col,
                      ((rows - 1) * kargs.stride_c + T::B_N) * sizeof(bf16_t));
        typename Engine::CTile c{};
        array<typename Engine::OperandFragments, T::DEPTH> queue;
        const int panels   = kargs.k / 128;
        auto load_operands = [&](int kblock) {
            return Engine::load_operands(g_a,
                                         g_b,
                                         g_sfa,
                                         g_sfb,
                                         kblock,
                                         kargs.stride_a,
                                         kargs.stride_b,
                                         kargs.stride_sfa,
                                         kargs.stride_sfb,
                                         row,
                                         col,
                                         kargs.m,
                                         kargs.n,
                                         lane_id,
                                         wave_id_m,
                                         wave_id_n);
        };

        // ===== Prologue =====
        static_for<T::DEPTH>([&](auto i) {
            if(decltype(i)::value < panels)
                queue[i] = load_operands(decltype(i)::value);
        });
        // ===== Main loop =====
        // Retire each slot before refilling it; the other slots retain their
        // outstanding loads. Static slots avoid register-array indexing in K.
#pragma clang loop unroll(disable)
        for(int base = 0; base < panels; base += T::DEPTH)
        {
            static_for<T::DEPTH>([&](auto i) {
                constexpr int slot = decltype(i)::value;
                if(base + slot < panels)
                {
                    Engine::compute_mfma(queue[slot], c);
                    __builtin_amdgcn_sched_barrier(0);
                    if(base + slot + T::DEPTH < panels)
                        queue[slot] = load_operands(base + slot + T::DEPTH);
                    __builtin_amdgcn_sched_barrier(0);
                }
            });
        }
        // ===== Epilogue =====
        // The dynamic queue has drained; bind each C fragment to its output layout.
        auto output_layout = [&](auto mi, auto ni) {
            constexpr int m = decltype(mi)::value;
            constexpr int n = decltype(ni)::value;
            return layout::make_layout_rc<false, T::B_M, T::B_N>((wave_id_m * T::E_M + m) * 16,
                                                                 (n * T::T_N + wave_id_n) * 16,
                                                                 lane_id,
                                                                 kargs.stride_c);
        };

        // ===== Write back =====
        static_for<T::E_M>([&](auto mi) {
            constexpr int m = decltype(mi)::value;
            static_for<T::E_N>([&](auto ni) {
                constexpr int n = decltype(ni)::value;
                const auto u_rc = output_layout(mi, ni);
                store<4>(g_c, cast<bf16_t>(c[m * T::E_N + n]), u_rc);
            });
        });
    }
}
#endif

template <class Traits>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, Traits::WG_PER_CU) void
opus_gemm_scheduled_fragment_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 kargs)
{
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
    opus_gemm_blockscale_mxscale_bpreshuffle_scheduled_fragment<Traits>(kargs);
#endif
}

template <class Traits>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, Traits::WG_PER_CU) void
opus_gemm_fragment_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 kargs)
{
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
    opus_gemm_blockscale_mxscale_bpreshuffle_fragment<Traits>(kargs);
#endif
}
