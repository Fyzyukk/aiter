// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <opus/hip_minimal.hpp>
#include <opus/opus.hpp>
#include "opus_gemm_traits_a8w8_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_gfx950.cuh"
#include "opus_gemm_traits_mxfp8_tt_4wave_gfx950.cuh"

#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
namespace opus_gemm_mxfp8_tt_detail {
using opus::operator""_I;

// Describe TR8 as a real DS instruction so LLVM tracks its asynchronous
// result before register moves or dependent MFMAs during co-execution.
template<int Vec, int Offset = 0, class Mem>
__device__ __forceinline__ auto native_tr_load(Mem& mem, int byte_offset) {
    using namespace opus;
    using Result = typename Mem::template vector_type<Vec>;
    static_assert(sizeof(Result) == 8 && sizeof(typename Mem::scalar_type) == 1);
    auto* p = reinterpret_cast<OPUS_LDS_ADDR i32x2_t*>(mem.ptr + byte_offset + Offset);
    return __builtin_bit_cast(Result, __builtin_amdgcn_ds_read_tr8_b64_v2i32(p));
}

// Partial native operand reads preserve the original per-fragment chunk order.
template<class Fragment, int Vec, int Repeat, int ChunkBegin, int ChunkCount, bool Transpose, class Mem, class Layout>
__device__ __forceinline__ auto load_fragment_chunks(
    Mem& mem, const Layout& u, int element_offset) {
    using namespace opus;
    using Scalar = typename Mem::scalar_type;
    using L = remove_cvref_t<Layout>;
    using LT = layout_load_traits<L, Vec>;
    constexpr int elems = sizeof(Fragment) / sizeof(Scalar);
    constexpr int count = elems / Vec;
    static_assert(Mem::vector_size == 1 && elems % Vec == 0);
    static_assert(Repeat >= 0 && (Repeat + 1) * count <= LT::r_elem.value);
    constexpr bool static_layout = is_static_tuple_v<typename L::Shape>
        && is_static_tuple_v<typename L::Stride>;
    static_assert(ChunkBegin >= 0 && ChunkCount > 0 && ChunkBegin + ChunkCount <= count);
    vector_t<Scalar, ChunkCount * Vec> raw;
    static_for<ChunkCount>([&](auto chunk) {
        constexpr int i = Repeat * count + ChunkBegin + decltype(chunk)::value;
        auto read = [&]() {
            if constexpr (static_layout) {
                using Shift = layout_shift_traits<L>;
                constexpr int off = layout_imm_offsets_v<L, Vec>[i] * sizeof(Scalar);
                if constexpr (off >= 0 && off <= 0xffff) {
                    const auto& base_layout = static_cast<const typename Shift::base&>(u);
                    const int base = base_layout(make_repeated_tuple(number<0>{},
                        number<size<decltype(LT::issue_space_vec)>()>{})) + element_offset;
                    if constexpr (Transpose)
                        return native_tr_load<Vec, off>(mem, base * sizeof(Scalar));
                    else
                        return mem.template load<Vec>(base + off / sizeof(Scalar));
                }
            }
            const auto offsets = layout_to_offsets<Vec>(u);
            if constexpr (Transpose)
                return native_tr_load<Vec>(mem, (offsets[i] + element_offset) * sizeof(Scalar));
            else
                return mem.template load<Vec>(offsets[i] + element_offset);
        };
        set_slice(raw, read(), number<decltype(chunk)::value * Vec>{},
            number<(decltype(chunk)::value + 1) * Vec>{});
    });
    return raw;
}

// Full prologue reads use the same chunk mapping as the co-execution path.
template<class Fragment, int Vec, int Repeat, bool Transpose, class Mem, class Layout>
__device__ __forceinline__ Fragment load_fragment(
    Mem& mem, const Layout& u, int element_offset) {
    constexpr int chunks = sizeof(Fragment) / (sizeof(typename Mem::scalar_type) * Vec);
    return load_fragment_chunks<Fragment, Vec, Repeat, 0, chunks, Transpose>(
        mem, u, element_offset);
}

// Source reads target already-retired fragments. Group the independent
// MFMA, LDSDMA, DS and address instructions inside one issue window.
template<int Mfmas, int Copies, int Reads, int Valus = 0>
__device__ __forceinline__ void schedule_step() {
    __builtin_amdgcn_sched_group_barrier(0x008, Mfmas, 0);
    if constexpr (Valus > 0)
        __builtin_amdgcn_sched_group_barrier(0x002, Valus, 0);
    if constexpr (Copies > 0)
        __builtin_amdgcn_sched_group_barrier(0x800, Copies, 0);
    if constexpr (Reads > 0)
        __builtin_amdgcn_sched_group_barrier(0x100, Reads, 0);
    // Compiler ordering only; no hardware stall is emitted.
    __builtin_amdgcn_sched_barrier(0);
}

// Keep each canonical operand fragment in AGPRs from its LDS read through MFMA.
template<class MMA>
struct pinned_ab_native_mma : MMA {
    using C = typename MMA::vtype_c;

    template<bool FirstK, class A, class B, class MI>
    __device__ static C issue(A& a, B& b, const C* current,
        int sfa, int sfb, MI mi) {
        static_assert(sizeof(A) == 32 && sizeof(B) == 32 && sizeof(C) == 16);
        auto raw_a = __builtin_bit_cast(opus::i32x8_t, a);
        auto raw_b = __builtin_bit_cast(opus::i32x8_t, b);
        // The empty constraint binds both complete 32-byte MFMA fragments.
        // Write the bound values back so later MFMAs reuse their AGPR storage.
        asm volatile("" : "+a"(raw_a), "+a"(raw_b));
        a = __builtin_bit_cast(A, raw_a);
        b = __builtin_bit_cast(B, raw_b);
        C result;
        auto compute = [&]() {
            if constexpr (FirstK)
                return MMA{}(raw_a, raw_b, sfa, sfb, mi, opus::number<0>{});
            else
                return MMA{}(raw_a, raw_b, *current, sfa, sfb, mi, opus::number<0>{});
        };
        result = compute();
        return result;
    }

    template<class A, class B, class MI, class NI>
    __device__ C operator()(A& a, B& b, int sfa, int sfb, MI mi, NI) const {
        return issue<true>(a, b, nullptr, sfa, sfb, mi);
    }

    template<class A, class B, class MI, class NI>
    __device__ C operator()(A& a, B& b, const C& c,
        int sfa, int sfb, MI mi, NI) const {
        return issue<false>(a, b, &c, sfa, sfb, mi);
    }
};

// Plain A[K,M] is loaded along contiguous M into K8 x M128 LDS slots.
template<class T>
__device__ inline auto make_layout_ga_scale(
    int lane_id, int wave_id_m, int wave_id_n, int stride_a) {
    constexpr int lanes_m = T::HALF_B_M / T::VEC_A;
    constexpr int lanes_k = T::WARP_SIZE / lanes_m;
    constexpr int repeats = T::B_K / (T::NUM_WAVES * lanes_k);
    // k = 32*repeat + 8*(2*wave_n + wave_m) + lane/8.
    constexpr auto shape = opus::make_tuple(
        opus::number<repeats>{}, opus::number<T::T_N>{},
        opus::number<T::T_M>{}, opus::number<lanes_k>{},
        opus::number<lanes_m>{}, opus::number<T::VEC_A>{});
    constexpr auto dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));
    return opus::make_layout<T::VEC_A>(shape,
        opus::unfold_x_stride(dim, shape, opus::tuple{stride_a, 1_I}),
        opus::unfold_p_coord(dim,
            opus::tuple{wave_id_n, wave_id_m, lane_id / lanes_m, lane_id % lanes_m}));
}

template<class T>
__device__ inline constexpr auto make_layout_sa_scale(int wave_id_m, int wave_id_n) {
    constexpr int P = T::smem_linear_wave + T::smem_padding;
    constexpr auto shape = opus::make_tuple(
        opus::number<T::smem_m_rep / T::NUM_WAVES>{},
        opus::number<T::T_N>{}, opus::number<T::T_M>{}, opus::number<T::VEC_A>{});
    constexpr auto dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}));
    // Original raw-A producer: slot = 4*repeat + wave, wave = 2*wave_n + wave_m.
    return opus::make_layout(shape,
        opus::unfold_x_stride(dim, shape, opus::tuple{opus::number<P>{}, 1_I}),
        opus::unfold_p_coord(dim, opus::tuple{wave_id_n, wave_id_m}));
}

template<class T>
__device__ inline constexpr auto make_layout_ra_scale(int lane_id, int wave_id_m) {
    constexpr int P = T::smem_linear_wave + T::smem_padding;
    constexpr int k_groups = T::WARP_SIZE / T::W_M;
    constexpr int k64_repeats = T::W_K / T::WARP_SIZE;
    constexpr int tr_reads_per_k16 = 16 / T::VEC_LDS_A;
    constexpr int source_k = T::VEC_LDS_A;
    constexpr int source_m_halves = T::W_M / T::VEC_LDS_A;
    // Original raw-A image: floor(k/8)*P + (k%8)*128 + m.
    // The TR8 source-lane permutation preserves chunks 0,8,64,72.
    constexpr auto shape = opus::make_tuple(
        opus::number<T::E_M>{}, opus::number<T::T_M>{},
        opus::number<k64_repeats>{}, opus::number<tr_reads_per_k16>{},
        opus::number<k_groups>{}, opus::number<source_k>{},
        opus::number<source_m_halves>{}, opus::number<T::VEC_LDS_A>{});
    constexpr auto dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}),
        opus::make_tuple(opus::y_dim{}),
        opus::make_tuple(opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));
    return opus::make_layout<T::VEC_LDS_A>(shape,
        opus::unfold_x_stride(dim, shape,
            opus::tuple{opus::number<T::W_M>{}, opus::number<8 * P>{},
                opus::number<P>{}, opus::number<2 * P>{},
                opus::number<T::HALF_B_M>{}, 1_I}),
        opus::unfold_p_coord(dim,
            opus::tuple{wave_id_m, lane_id / T::W_M,
                (lane_id % T::W_M) / source_m_halves, lane_id % source_m_halves}));
}

// Alternate plain-A producer: each LDS slot contains K64 x M16.
template<class T>
__device__ inline auto make_layout_ga_scale_single_slot(
    int lane_id, int wave_id_m, int wave_id_n, int stride_a) {
    constexpr int m_repeats = T::WARP_SIZE / 16 / T::T_M;
    constexpr int k_repeats = T::B_K / T::WARP_SIZE;
    // q = 4*m_repeat + 2*wave_m + wave_n; k = 64*k_half + lane.
    constexpr auto shape = opus::make_tuple(
        opus::number<m_repeats>{}, opus::number<T::T_N>{},
        opus::number<T::T_M>{}, opus::number<k_repeats>{},
        opus::number<T::WARP_SIZE>{}, opus::number<T::VEC_A>{});
    constexpr auto dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}));
    return opus::make_layout<T::VEC_A>(shape,
        opus::unfold_x_stride(dim, shape,
            opus::tuple{opus::number<T::W_M>{}, stride_a, 1_I}),
        opus::unfold_p_coord(dim, opus::tuple{wave_id_m, wave_id_n, lane_id}));
}

template<class T>
__device__ inline constexpr auto make_layout_sa_scale_single_slot(int wave_id_m, int wave_id_n) {
    constexpr int P = T::smem_linear_wave + T::smem_padding;
    constexpr auto shape = opus::make_tuple(
        opus::number<T::smem_m_rep / T::NUM_WAVES>{},
        opus::number<T::T_N>{}, opus::number<T::T_M>{}, opus::number<T::VEC_A>{});
    constexpr auto dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}));
    // slot = 4*repeat + 2*wave_m + wave_n; repeat = 2*m_repeat + k_half.
    return opus::make_layout(shape,
        opus::unfold_x_stride(dim, shape, opus::tuple{opus::number<P>{}, 1_I}),
        opus::unfold_p_coord(dim, opus::tuple{wave_id_m, wave_id_n}));
}

template<class T>
__device__ inline constexpr auto make_layout_ra_scale_single_slot(int lane_id, int wave_id_m) {
    constexpr int P = T::smem_linear_wave + T::smem_padding;
    constexpr int k_groups = T::WARP_SIZE / T::W_M;
    constexpr int k64_repeats = T::W_K / T::WARP_SIZE;
    constexpr int tr_reads_per_k16 = 16 / T::VEC_LDS_A;
    constexpr int m_repeats = T::WARP_SIZE / 16 / T::T_M;
    // Source = 8P*(em/2) + P*(2*(em%2)+wave_m) + 4P*hi
    //        + 256*(lane/16) + 128*lo + 8*(lane%16).
    constexpr auto shape = opus::make_tuple(
        opus::number<T::E_M / m_repeats>{}, opus::number<m_repeats>{},
        opus::number<T::T_M>{}, opus::number<k64_repeats>{},
        opus::number<k_groups>{}, opus::number<tr_reads_per_k16>{},
        opus::number<T::W_M>{}, opus::number<T::VEC_LDS_A>{});
    constexpr auto dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}),
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}),
        opus::make_tuple(opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));
    return opus::make_layout<T::VEC_LDS_A>(shape,
        opus::unfold_x_stride(dim, shape,
            opus::tuple{opus::number<8 * P>{}, opus::number<P>{},
                opus::number<4 * P>{}, opus::number<T::W_M * 16>{}, 1_I}),
        opus::unfold_p_coord(dim,
            opus::tuple{wave_id_m, lane_id / T::W_M, lane_id % T::W_M}));
}

// B is shuffle_weight(B[N,K], layout=(16,16)). These three layouts are
// the current NT packed producer/LDS/reader path, without AGPR pinning.
template<class T>
__device__ inline constexpr auto make_layout_gb_scale(
    int lane_id, int wave_id_m, int wave_id_n, int stride_b) {
    constexpr int lanes_n = T::W_N;
    constexpr int lanes_k = T::WARP_SIZE / lanes_n;
    constexpr int k_per_wave_load = lanes_k * T::VEC_B;
    constexpr int n_repeats = T::HALF_B_N / (T::NUM_WAVES * lanes_n);
    constexpr int k_repeats = T::B_K / k_per_wave_load;

    constexpr auto shape = opus::make_tuple(
        opus::number<n_repeats>{}, 
        opus::number<T::T_N>{},
        opus::number<T::T_M>{}, 
        opus::number<k_repeats>{},
        opus::number<T::WARP_SIZE>{}, 
        opus::number<T::VEC_B>{});

    constexpr auto dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout<T::VEC_B>(shape,
        opus::unfold_x_stride(dim, shape, opus::tuple{T::W_N * stride_b, 1_I}),
        opus::unfold_p_coord(dim, opus::tuple{wave_id_m, wave_id_n, lane_id}));
}

template<class T>
__device__ inline constexpr auto make_layout_sb_scale(int wave_id_m, int wave_id_n) {

    constexpr auto shape = opus::make_tuple(
        opus::number<T::smem_n_rep / T::NUM_WAVES>{},
        opus::number<T::T_N>{},
        opus::number<T::T_M>{}, 
        opus::number<T::VEC_B>{});

    constexpr auto dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}));

    return opus::make_layout(shape,
        opus::unfold_x_stride(dim, shape, opus::tuple{opus::number<T::smem_linear_wave + T::smem_padding>{}, 1_I}),
        opus::unfold_p_coord(dim, opus::tuple{wave_id_m, wave_id_n}));
}

template<class T>
__device__ inline constexpr auto make_layout_rb_scale(int lane_id, int wave_id_n) {
    constexpr int P = T::smem_linear_wave + T::smem_padding;
    constexpr int n_repeats = T::NUM_WAVES / T::T_N;

    constexpr auto shape = opus::make_tuple(
        opus::number<T::E_N / n_repeats>{}, 
        opus::number<n_repeats>{},
        opus::number<T::T_N>{},
        opus::number<T::W_N * T::W_K / T::WARP_SIZE / T::VEC_B>{},
        opus::number<T::E_K>{}, 
        opus::number<T::WARP_SIZE>{}, 
        opus::number<T::VEC_B>{});

    constexpr auto dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}),
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}),
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout<T::VEC_B>(shape,
        opus::unfold_x_stride(dim, shape, opus::tuple{opus::number<8 * P>{}, opus::number<P>{}, opus::number<4 * P>{}, 1_I}),
        opus::unfold_p_coord(dim, opus::tuple{wave_id_n, lane_id}));
}

template<class T>
__device__ inline constexpr auto make_layout_gsfa_scale(int lane_id, int wave_id_m, int wave_id_n, int stride_sfa) {
    constexpr auto block_shape = opus::make_tuple(
        opus::number<T::SFA_PASSES_PER_CACHE_PANEL>{}, // 4 cache-fill passes
        opus::number<T::T_N>{}, // 2
        opus::number<T::T_M>{}, // 2
        opus::number<T::SFA_K_COLUMNS_PER_WAVE>{}, // 4
        opus::number<T::B_M / T::VEC_SCALE_A>{}, // 16 lanes per K column
        opus::number<T::VEC_SCALE_A>{}); // 16 contiguous M elements per lane

    constexpr auto block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    // M-vector order for quad packing: 0,2,4,6,1,3,5,7,8,10,12,14,9,11,13,15.
    const int lane_id_m = (lane_id & 8) | ((lane_id & 3) << 1) | ((lane_id & 4) >> 2);
    return opus::make_layout<T::VEC_SCALE_A>(
        block_shape,
        opus::unfold_x_stride(block_dim, block_shape, opus::tuple{stride_sfa, 1_I}),
        opus::unfold_p_coord(block_dim, opus::tuple{wave_id_n, wave_id_m, lane_id / T::W_M, lane_id_m}));
}

template<class T>
__device__ inline constexpr auto make_layout_ssfa_scale(int lane_id, int wave_id_m, int wave_id_n) {
    constexpr auto block_shape = opus::make_tuple(
        opus::number<T::SFA_PASSES_PER_CACHE_PANEL>{}, // 4 cache-fill passes
        opus::number<T::T_N>{}, // 2
        opus::number<T::T_M>{}, // 2
        opus::number<T::SFA_K_COLUMNS_PER_WAVE>{}, // 4
        opus::number<T::B_M / T::VEC_SF>{}, // 64 packed M dwords per K column
        opus::number<T::VEC_SF>{}); // 4 M-repeat scale bytes per dword

    constexpr auto block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    // First packed dword for this lane; successive transposed words are 8 dwords apart.
    const int lane_id_m = ((lane_id & 4) << 3) | ((lane_id & 3) << 1) | ((lane_id & 8) >> 3);
    return opus::make_layout<T::VEC_SF>(
        block_shape,
        opus::unfold_x_stride(block_dim, block_shape, opus::tuple{opus::number<T::SFA_PANEL_PITCH>{}, 1_I}),
        opus::unfold_p_coord(block_dim, opus::tuple{wave_id_n, wave_id_m, lane_id / T::W_M, lane_id_m}));
}

template<class T, int Vec = T::VEC_SCALE_SF>
__device__ inline constexpr auto make_layout_rsfa_scale(int lane_id, int wave_id_m) {
    static_assert(Vec == T::VEC_SCALE_SF || Vec == T::VEC_SCALE_SF_PAIR);

    constexpr auto block_shape = opus::make_tuple(
        opus::number<T::T_M>{}, // 2 wave_m
        opus::number<T::W_M>{}, // 16 M lanes
        opus::number<Vec>{});

    constexpr auto block_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}));

    return opus::make_layout<Vec>(
        block_shape,
        opus::unfold_x_stride(block_dim, block_shape, opus::tuple{opus::number<T::VEC_SCALE_SF_PAIR>{}, 1_I}),
        opus::unfold_p_coord(block_dim, opus::tuple{wave_id_m, lane_id % T::W_M}));
}

// TT scaleB is contiguous [N128,K128]; waves 0/1 publish one N128 half each.
template<class T>
__device__ inline constexpr auto make_layout_gsfb_scale(int lane_id, int wave_id, int stride_sfb) {
    constexpr auto block_shape = opus::make_tuple(
        opus::number<T::SCALE_N_HALVES>{},
        opus::number<T::SCALE_PANEL_K_CAPACITY>{},
        opus::number<1>{});
    constexpr auto block_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));
    return opus::make_layout<1>(
        block_shape,
        opus::unfold_x_stride(block_dim, block_shape, opus::tuple{stride_sfb, 1_I}),
        opus::unfold_p_coord(block_dim, opus::tuple{wave_id, lane_id}));
}

template<class T>
__device__ inline constexpr auto make_layout_ssfb_scale(int lane_id, int wave_id) {
    constexpr auto block_shape = opus::make_tuple(
        opus::number<T::SCALE_N_HALVES>{}, // 2 N128 halves
        opus::number<T::SCALE_PANEL_K_CAPACITY>{}, // 64 K128 slots
        opus::number<T::VEC_SF>{}); // low byte holds the N128 scale; op_sel_B is 0

    constexpr auto block_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}));

    return opus::make_layout<T::VEC_SF>(
        block_shape,
        opus::unfold_x_stride(block_dim, block_shape, opus::tuple{opus::number<T::VEC_SF>{}, opus::number<T::SCALE_N_HALVES * T::VEC_SF>{}, 1_I}),
        opus::unfold_p_coord(block_dim, opus::tuple{wave_id, lane_id}));
}

template<class T, int Vec = T::VEC_SCALE_SF>
__device__ inline constexpr auto make_layout_rsfb_scale() {
    static_assert(Vec == T::VEC_SCALE_SF || Vec == T::VEC_SCALE_SF_PAIR);

    constexpr auto block_shape = opus::make_tuple(opus::number<Vec>{});
    constexpr auto block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}));

    return opus::make_layout<Vec>(
        block_shape,
        opus::unfold_x_stride(block_dim, block_shape, opus::tuple{1_I}),
        opus::unfold_p_coord(block_dim, opus::tuple{}));
}

} // namespace opus_gemm_mxfp8_tt_detail

// TT: plain A[K,M] is transposed while reading LDS; B[N,K] uses the fixed
// shuffle_weight(layout=(16,16)) packed NT path with ordinary LDS reads.
template<class Traits>
__device__ __forceinline__ void opus_gemm_mxfp8_tt_4wave(
    opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 args) {
    using namespace opus;
    using namespace opus_gemm_mxfp8_tt_detail;
    using T = opus::remove_cvref_t<Traits>;
    static_assert(T::BLOCK_SIZE == 256 && T::B_M == 256 && T::B_N == 256 && T::B_K == 128);
    static_assert(T::T_M == 2 && T::T_N == 2 && T::T_K == 1 && T::E_M == 4 && T::E_N == 4);
    static_assert(T::VEC_A == 16 && T::VEC_B == 16 && T::VEC_LDS_A == 8 && T::VEC_LDS_B == 16);

    const int lane = thread_id_x() % 64;
    const int wave = __builtin_amdgcn_readfirstlane(thread_id_x() / 64);
    const int wave_m = wave % 2;
    const int wave_n = wave / 2;
    int block_m = block_id_y();
    int block_n = block_id_x();
    if (((args.m | args.n) & 511) == 0) {
        const int old_m = block_m;
        block_m = (block_m & ~1) | (block_n & 1);
        block_n = (block_n & ~1) | (old_m & 1);
    }
    const int row = block_m * 256;
    const int col = block_n * 256;
    const int batch = block_id_z();

    auto ga = make_gmem(reinterpret_cast<const fp8_t*>(args.ptr_a) +
        static_cast<int64_t>(batch) * args.stride_a_batch + row,
        static_cast<unsigned>(int64_t(args.k) * args.stride_a - row));
    auto gb = make_gmem(reinterpret_cast<const fp8_t*>(args.ptr_b) +
        static_cast<int64_t>(batch) * args.stride_b_batch + static_cast<int64_t>(col) * args.stride_b,
        static_cast<unsigned>(int64_t(args.n - col) * args.stride_b));
    auto gc = make_gmem(reinterpret_cast<bf16_t*>(args.ptr_c) +
        static_cast<int64_t>(batch) * args.stride_c_batch + static_cast<int64_t>(row) * args.stride_c + col);
    using D_SF = unsigned char;
    using D_SF_PACK = unsigned;
    auto gsfa = make_gmem(reinterpret_cast<const D_SF*>(args.ptr_sfa) +
        static_cast<int64_t>(batch) * args.stride_sfa_batch + row,
        static_cast<unsigned>(args.stride_sfa_batch - row));
    const int64_t sfb_offset = int64_t(col / T::GROUP_N) * args.stride_sfb;
    auto gsfb = make_gmem(reinterpret_cast<const D_SF*>(args.ptr_sfb) +
        static_cast<int64_t>(batch) * args.stride_sfb_batch + sfb_offset,
        static_cast<unsigned>(args.stride_sfb_batch - sfb_offset));

    constexpr int a_half_elems = T::SMEM_HALF_A_ELEMS;
    constexpr int b_half_elems = T::SMEM_HALF_B_ELEMS;
    alignas(16) __shared__ unsigned char matrix_lds[T::SMEM_A_ELEMS + T::SMEM_B_ELEMS];
    alignas(16) __shared__ D_SF sfa_lds[T::SFA_PANEL_BYTES];
    alignas(16) __shared__ D_SF sfb_lds[T::SFB_PANEL_BYTES];
    auto ssfa = make_smem(sfa_lds);
    auto ssfb = make_smem(sfb_lds);
    auto sa = make_smem(reinterpret_cast<fp8_t*>(matrix_lds));
    auto sb = make_smem(reinterpret_cast<fp8_t*>(matrix_lds + T::SMEM_A_ELEMS));
    static_assert(T::LDS_BYTES <= 163840);

    using D_A = fp8_t;
    using D_B = fp8_t;
    using D_C = bf16_t;
    using D_ACC = fp32_t;
    auto mma = make_tiled_mma<D_A, D_B, D_ACC>(seq<T::E_M, T::E_N, T::E_K>{},
        seq<T::T_M, T::T_N, T::T_K>{}, seq<T::W_M, T::W_N, T::W_K>{}, mfma_adaptor_swap_ab{});
    using BaseMMA = pinned_ab_native_mma<
        typename decltype(mma)::MMA>;
    using AFragment = typename BaseMMA::vtype_a;
    using BFragment = typename BaseMMA::vtype_b;
    using BRegs = opus::array<BFragment, T::E_N>;
    using CRegs = typename decltype(mma)::vtype_c;
    constexpr int ELEM_C = BaseMMA::elem_c;
    static_assert(sizeof(CRegs) == T::E_M * T::E_N * ELEM_C * sizeof(D_ACC));
    using ARegs = opus::array<AFragment, T::E_M>;
    ARegs v_a[2];
    BRegs v_b[2];
    opus::array<typename BaseMMA::vtype_c, T::E_M * T::E_N> v_c[2][2];

    // OPUS layouts describe each operand from global memory through LDS to
    // registers. Only tile/half/stage offsets vary inside the mainloop.
    // Coalesced A, K8 x M128 slots. Comment all six declarations to switch.
    const auto u_ga = make_layout_ga_scale<T>(lane, wave_m, wave_n, args.stride_a);
    const auto u_gb = make_layout_gb_scale<T>(lane, wave_m, wave_n, args.stride_b);
    const auto u_sa = make_layout_sa_scale<T>(wave_m, wave_n);
    const auto u_sb = make_layout_sb_scale<T>(wave_m, wave_n);
    const auto u_ra = make_layout_ra_scale<T>(lane, wave_m);
    const auto u_rb = make_layout_rb_scale<T>(lane, wave_n);
    // Single-slot A, K64 x M16 slots. Uncomment this complete block to switch.
    /*
    const auto u_ga = make_layout_ga_scale_single_slot<T>(lane, wave_m, wave_n, args.stride_a);
    const auto u_gb = make_layout_gb_scale<T>(lane, wave_m, wave_n, args.stride_b);
    const auto u_sa = make_layout_sa_scale_single_slot<T>(wave_m, wave_n);
    const auto u_sb = make_layout_sb_scale<T>(wave_m, wave_n);
    const auto u_ra = make_layout_ra_scale_single_slot<T>(lane, wave_m);
    const auto u_rb = make_layout_rb_scale<T>(lane, wave_n);
    */
    auto ga_offset = [&](int half, int tile) { return tile * 128 * args.stride_a + half * 128; };
    auto gb_offset = [&](int half, int tile) { return half * 128 * args.stride_b + tile * 128 * 16; };
    auto sa_offset = [&](int stage, int half) { return (stage * 2 + half) * a_half_elems; };
    auto sb_offset = [&](int stage, int half) { return (stage * 2 + half) * b_half_elems; };
    const auto ga_copy_offsets = layout_to_offsets<T::VEC_A>(u_ga);
    const auto gb_copy_offsets = layout_to_offsets<T::VEC_B>(u_gb);
    const auto sa_copy_offsets = layout_to_offsets<T::VEC_A>(u_sa);
    const auto sb_copy_offsets = layout_to_offsets<T::VEC_B>(u_sb);
    // Spread each half's four copies between independent MFMA pairs.
    auto prefetch_copy = [&](int tile, int stage, auto part_i, auto pass_i) {
        constexpr int part = decltype(part_i)::value;
        constexpr int pass = decltype(pass_i)::value;
        constexpr int half = part % 2;
        if constexpr (part < 2)
            async_load<T::VEC_A>(ga,
                reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(sa.ptr)
                    + sa_copy_offsets[pass] + sa_offset(stage, half)),
                ga_copy_offsets[pass], ga_offset(half, tile));
        else
            async_load<T::VEC_B>(gb,
                reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(sb.ptr)
                    + sb_copy_offsets[pass] + sb_offset(stage, half)),
                gb_copy_offsets[pass], gb_offset(half, tile));
    };

    // Cache the same 64 K128 columns as NT. SFA is packed in VGPRs
    // before publication; consumers read both M/N halves in one LDS pair.
    const int loops = args.k / T::B_K;
    constexpr int scale_panel_mask = T::SCALE_PANEL_K_CAPACITY - 1;
    static_assert((T::SCALE_PANEL_K_CAPACITY & scale_panel_mask) == 0);
    const auto u_gsfa = make_layout_gsfa_scale<T>(lane, wave_m, wave_n, args.stride_sfa);
    const auto u_ssfa = make_layout_ssfa_scale<T>(lane, wave_m, wave_n);
    const auto u_rsfa = make_layout_rsfa_scale<T, T::VEC_SCALE_SF_PAIR>(lane, wave_m);
    const auto u_gsfb = make_layout_gsfb_scale<T>(lane, wave, args.stride_sfb);
    const auto u_ssfb = make_layout_ssfb_scale<T>(lane, wave);
    const auto u_rsfb = make_layout_rsfb_scale<T, T::VEC_SCALE_SF_PAIR>();
    const auto sfa_gmem_offsets = layout_to_offsets<T::VEC_SCALE_A>(u_gsfa);
    const auto sfa_smem_offsets = layout_to_offsets<T::VEC_SF>(u_ssfa);
    constexpr int sfa_lds_word_stride = T::VEC_SF * (T::B_M / T::HALF_B_M) * T::E_M;
    const int lane_in_quad = lane % 4;
    const unsigned pair_byte_select = (lane_in_quad % 2) ? 0x03070105u : 0x06020400u;
    const unsigned quad_byte_select = (lane_in_quad / 2) ? 0x03020706u : 0x05040100u;

    auto fill_scale_panel = [&](int panel_begin) {
        vector_t<D_SF, T::VEC_SCALE_A> sfa_panel_raw[T::SFA_PASSES_PER_CACHE_PANEL];
        static_for<T::SFA_PASSES_PER_CACHE_PANEL>([&](auto pass_i) {
            constexpr int pass = decltype(pass_i)::value;
            const int first_k_group = panel_begin + pass * T::SFA_K_COLUMNS_PER_PASS
                + wave * T::SFA_K_COLUMNS_PER_WAVE;
            sfa_panel_raw[pass] = {};
            // The bounded resource returns zero for the final partial pass's
            // K columns past loops, without reading into the next batch.
            if (first_k_group < loops)
                sfa_panel_raw[pass] = load<T::VEC_SCALE_A>(gsfa,
                    sfa_gmem_offsets[pass] + panel_begin * args.stride_sfa);
        });
        D_SF_PACK raw_b = 0;
        if (wave < T::SCALE_N_HALVES && panel_begin + lane < loops)
            raw_b = static_cast<D_SF_PACK>(load<1>(gsfb, u_gsfb + panel_begin)[0]);

        // Global reads must finish before DPP/byte permutation and LDS stores.
        s_waitcnt_vmcnt(number<0>{});
        static_for<T::SFA_PASSES_PER_CACHE_PANEL>([&](auto pass_i) {
            constexpr int pass = decltype(pass_i)::value;
            const int first_k_group = panel_begin + pass * T::SFA_K_COLUMNS_PER_PASS
                + wave * T::SFA_K_COLUMNS_PER_WAVE;
            if (first_k_group < loops) {
                const auto raw_words = __builtin_bit_cast(vector_t<D_SF_PACK, 4>, sfa_panel_raw[pass]);
                static_for<4>([&](auto word_i) {
                    constexpr int word = decltype(word_i)::value;
                    const D_SF_PACK adjacent = mov_dpp(raw_words[word], number<0xb1>{});
                    const D_SF_PACK paired = __builtin_amdgcn_perm(adjacent, raw_words[word], pair_byte_select);
                    const D_SF_PACK opposite = mov_dpp(paired, number<0x4e>{});
                    const D_SF_PACK packed = __builtin_amdgcn_perm(opposite, paired, quad_byte_select);
                    store<T::VEC_SF>(ssfa,
                        __builtin_bit_cast(vector_t<D_SF, T::VEC_SF>, packed),
                        sfa_smem_offsets[pass] + word * sfa_lds_word_stride);
                });
            }
        });
        if (wave < T::SCALE_N_HALVES)
            store<T::VEC_SF>(ssfb,
                __builtin_bit_cast(vector_t<D_SF, T::VEC_SF>, raw_b), u_ssfb);
        s_waitcnt_lgkmcnt(number<0>{});
        sync_threads();
    };

    vector_t<D_SF_PACK, 2> v_sfa, v_sfb;
    auto load_scales = [&](int tile) {
        v_sfa = __builtin_bit_cast(vector_t<D_SF_PACK, 2>,
            load<T::VEC_SCALE_SF_PAIR>(ssfa,
                u_rsfa + (tile & scale_panel_mask) * T::SFA_PANEL_PITCH));
        v_sfb = __builtin_bit_cast(vector_t<D_SF_PACK, 2>,
            load<T::VEC_SCALE_SF_PAIR>(ssfb,
                u_rsfb + (tile & scale_panel_mask) * T::SCALE_N_HALVES * T::VEC_SF));
    };

    // Ki computes from registers, Ki+1 is read from one LDS slot, and
    // Ki+2 is copied into the other. Prime both slots before computing.
    static_for<4>([&](auto part) {
        static_for<4>([&](auto pass) { prefetch_copy(0, 0, part, pass); });
    });
    if (loops > 1) {
        static_for<4>([&](auto part) {
            static_for<4>([&](auto pass) { prefetch_copy(1, 1, part, pass); });
        });
    }
    fill_scale_panel(0);

    static_for<2>([&](auto half_i) {
        constexpr int half = decltype(half_i)::value;
        static_for<T::E_M>([&](auto mi_i) {
            constexpr int mi = decltype(mi_i)::value;
            v_a[half][mi] = load_fragment<AFragment, T::VEC_LDS_A, mi, true>(
                sa, u_ra, sa_offset(0, half));
        });
        static_for<T::E_N>([&](auto ni_i) {
            constexpr int ni = decltype(ni_i)::value;
            v_b[half][ni] = load_fragment<BFragment, T::VEC_LDS_B, ni, false>(
                sb, u_rb, sb_offset(0, half));
        });
    });
    load_scales(0);
    // Every wave must finish reading K0 before its LDS slot can hold K2.
    s_waitcnt_lgkmcnt(number<0>{});
    sync_threads();
    int stage = 0;

    // ===== Main loop =====
    // One K128 iteration: MFMA K_i, read K_i+1, and prefetch K_i+2.
    auto main_loop_iteration = [&](int tile, auto first_tile, auto next_tile, auto later_tile,
        OPUS_LDS_ADDR char* __restrict__ read_a,
        OPUS_LDS_ADDR char* __restrict__ read_b,
        OPUS_LDS_ADDR char* __restrict__ write_a,
        OPUS_LDS_ADDR char* __restrict__ write_b) {
        constexpr bool first_k = decltype(first_tile)::value;
        auto next_sa = sa;
        auto next_sb = sb;
        next_sa.ptr = read_a;
        next_sb.ptr = read_b;
        const int future_tile = tile + 2;
        auto prefetch = [&](auto part_i, auto pass_i) {
            if constexpr (first_k) {
                if (loops < 3) return;
            }
            constexpr int part = decltype(part_i)::value;
            constexpr int pass = decltype(pass_i)::value;
            constexpr int half = part % 2;
            if constexpr (part < 2)
                async_load<T::VEC_A>(ga,
                    reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(write_a)
                        + sa_copy_offsets[pass] + sa_offset(0, half)),
                    ga_copy_offsets[pass], ga_offset(half, future_tile));
            else
                async_load<T::VEC_B>(gb,
                    reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(write_b)
                        + sb_copy_offsets[pass] + sb_offset(0, half)),
                    gb_copy_offsets[pass], gb_offset(half, future_tile));
        };
        constexpr bool has_next = decltype(next_tile)::value;
        constexpr bool has_later = decltype(later_tile)::value;
        constexpr int a_chunks = sizeof(AFragment) / T::VEC_LDS_A;
        constexpr int b_chunks = sizeof(BFragment) / T::VEC_LDS_B;
        if (tile != 0 && (tile & scale_panel_mask) == 0) {
            fill_scale_panel(tile);
            load_scales(tile);
            s_waitcnt_lgkmcnt(number<0>{});
            __builtin_amdgcn_sched_barrier(0);
        }
        const D_SF_PACK sfa[2] = {v_sfa[0], v_sfa[1]};
        const D_SF_PACK sfb[2] = {v_sfb[0], v_sfb[1]};
        __builtin_amdgcn_sched_barrier(0);

        auto mfma = [&](auto hm, auto hn, auto mi, auto ni) {
            constexpr int index = decltype(mi)::value * T::E_N + decltype(ni)::value;
            if constexpr (first_k)
                v_c[hm][hn][index] = BaseMMA{}(v_a[hm][mi], v_b[hn][ni],
                    static_cast<int>(sfa[hm]), static_cast<int>(sfb[hn]), mi, number<0>{});
            else
                v_c[hm][hn][index] = BaseMMA{}(v_a[hm][mi], v_b[hn][ni], v_c[hm][hn][index],
                    static_cast<int>(sfa[hm]), static_cast<int>(sfb[hn]), mi, number<0>{});
        };
        auto load_a = [&](auto half_i, auto repeat_i, auto begin_i, auto count_i) {
            if constexpr (first_k) {
                if (loops < 2) return;
            }
            constexpr int half = decltype(half_i)::value, repeat = decltype(repeat_i)::value;
            constexpr int begin = decltype(begin_i)::value, count = decltype(count_i)::value;
            // Ki+1 scales are read here while MFMA still uses the Ki snapshot.
            if constexpr (half == 0 && repeat == 3 && begin == 0) {
                if (((tile + 1) & scale_panel_mask) != 0) load_scales(tile + 1);
            }
            set_slice(v_a[half][repeat],
                load_fragment_chunks<AFragment, T::VEC_LDS_A, repeat, begin, count, true>(
                    next_sa, u_ra, sa_offset(0, half)),
                number<begin * T::VEC_LDS_A>{}, number<(begin + count) * T::VEC_LDS_A>{});
        };
        auto load_b = [&](auto half_i, auto repeat_i, auto begin_i, auto count_i) {
            if constexpr (first_k) {
                if (loops < 2) return;
            }
            constexpr int half = decltype(half_i)::value, repeat = decltype(repeat_i)::value;
            constexpr int begin = decltype(begin_i)::value, count = decltype(count_i)::value;
            set_slice(v_b[half][repeat],
                load_fragment_chunks<BFragment, T::VEC_LDS_B, repeat, begin, count, false>(
                    next_sb, u_rb, sb_offset(0, half)),
                number<begin * T::VEC_LDS_B>{}, number<(begin + count) * T::VEC_LDS_B>{});
        };
        // C[0][0] and C[0][1]: consume A0 across both opposite halves.
        // Each retired fragment is reloaded one DS chunk per MFMA pair.
        static_for<4>([&](auto mi) {
            static_for<4>([&](auto ni) {
                constexpr int m = decltype(mi)::value, n = decltype(ni)::value;
                mfma(number<0>{}, number<0>{}, mi, ni);
                if constexpr (has_next && m > 0)
                    load_a(number<0>{}, number<m - 1>{},
                        ni, number<1>{});
                schedule_step<1, 0, (has_next && m > 0 ? 1 : 0),
                    has_next && m == 1 && n == 0 ? 1 : 0>();
                mfma(number<0>{}, number<1>{}, mi, ni);
                if constexpr (has_later) prefetch(mi, ni);
                schedule_step<1, has_later ? 1 : 0, 0>();
            });
        });

        // C[1][0]: finish A0 reloads, then reload retired B0 fragments.
        static_for<4>([&](auto ni) {
            static_for<4>([&](auto mi) {
                constexpr int m = decltype(mi)::value, n = decltype(ni)::value;
                constexpr int begin = m * b_chunks / 4;
                constexpr int end = (m + 1) * b_chunks / 4;
                mfma(number<1>{}, number<0>{}, mi, ni);
                if constexpr (has_next && n == 0)
                    load_a(number<0>{}, number<3>{},
                        mi, number<1>{});
                if constexpr (has_next && n > 0 && end > begin)
                    load_b(number<0>{}, number<n - 1>{},
                        number<begin>{}, number<end - begin>{});
                schedule_step<1, 0,
                    has_next ? (n == 0 ? (m == 0 ? 3 : 1) : end - begin) : 0>();
            });
        });

        // C[1][1]: retire one row/column pair at a time and load its next fragments.
        static_for<4>([&](auto diagonal_i) {
            constexpr int diagonal = decltype(diagonal_i)::value;
            constexpr int rows = 4 - diagonal, columns = 3 - diagonal;
            static_for<rows>([&](auto step_i) {
                constexpr int step = decltype(step_i)::value;
                constexpr int begin = step * b_chunks / rows;
                constexpr int end = (step + 1) * b_chunks / rows;
                if constexpr (has_next && diagonal == 3) {
                    // B1[2] has retired; its read can precede the last MFMA.
                    load_b(number<1>{}, number<2>{},
                        number<0>{}, number<b_chunks>{});
                    __builtin_amdgcn_sched_group_barrier(0x100, b_chunks, 0);
                    __builtin_amdgcn_sched_barrier(0);
                }
                mfma(number<1>{}, number<1>{}, number<diagonal>{},
                    number<diagonal + step>{});
                if constexpr (has_next && diagonal > 0 && diagonal < 3 && end > begin)
                    load_b(number<1>{}, number<diagonal - 1>{},
                        number<begin>{}, number<end - begin>{});
                if constexpr (has_next && diagonal == 0 && step < b_chunks)
                    load_b(number<0>{}, number<3>{},
                        number<step>{}, number<1>{});
                schedule_step<1, 0, has_next ? (diagonal > 0 ?
                    (diagonal == 3 ? 0 : end - begin) : (step < b_chunks ? 1 : 0)) : 0>();
            });
            static_for<columns>([&](auto step_i) {
                constexpr int step = decltype(step_i)::value;
                constexpr int begin = step * a_chunks / columns;
                constexpr int end = (step + 1) * a_chunks / columns;
                mfma(number<1>{}, number<1>{}, number<diagonal + 1 + step>{},
                    number<diagonal>{});
                if constexpr (has_next && end > begin)
                    load_a(number<1>{}, number<diagonal>{},
                        number<begin>{}, number<end - begin>{});
                schedule_step<1, 0, has_next ? end - begin : 0>();
            });
            if constexpr (has_next && diagonal == 3) {
                load_a(number<1>{}, number<3>{},
                    number<0>{}, number<a_chunks>{});
                load_b(number<1>{}, number<3>{},
                    number<0>{}, number<b_chunks>{});
            }
        });

        if constexpr (has_next) {
            // Finish every wave's Ki+1 reads before its slot holds Ki+3.
            // Publish Ki+2 here while preserving the original copy lead time.
            if (!first_k || loops > 1) {
                s_waitcnt_lgkmcnt(number<0>{});
                if constexpr (has_later) {
                    if (!first_k || loops > 2) s_waitcnt_vmcnt(number<0>{});
                }
                sync_threads();
                __builtin_amdgcn_sched_barrier(0);
            }
        }
        stage ^= 1;
    };
    // Initialize every C fragment with its first MFMA's literal C=0.
    main_loop_iteration(0, number<true>{}, number<true>{}, number<true>{},
        sa.ptr + sa_offset(1, 0), sb.ptr + sb_offset(1, 0),
        sa.ptr + sa_offset(0, 0), sb.ptr + sb_offset(0, 0));
    int tile = 1;
#pragma clang loop unroll(disable)
    for (; tile + 2 < loops; ++tile)
        main_loop_iteration(tile, number<false>{}, number<true>{}, number<true>{},
            sa.ptr + sa_offset(stage ^ 1, 0), sb.ptr + sb_offset(stage ^ 1, 0),
            sa.ptr + sa_offset(stage, 0), sb.ptr + sb_offset(stage, 0));
    if (tile + 1 < loops) {
        main_loop_iteration(tile, number<false>{}, number<true>{}, number<false>{},
            sa.ptr + sa_offset(stage ^ 1, 0), sb.ptr + sb_offset(stage ^ 1, 0),
            sa.ptr + sa_offset(stage, 0), sb.ptr + sb_offset(stage, 0));
        ++tile;
    }
    if (tile < loops)
        main_loop_iteration(tile, number<false>{}, number<false>{}, number<false>{},
            sa.ptr + sa_offset(stage ^ 1, 0), sb.ptr + sb_offset(stage ^ 1, 0),
            sa.ptr + sa_offset(stage, 0), sb.ptr + sb_offset(stage, 0));

    // Preserve the validated accumulator-to-epilogue scheduling gap.
    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_nop 7" : : : "memory");
    __builtin_amdgcn_sched_barrier(0);

    // Pair neighboring C fragments across lane groups for 16-byte stores.
    const int store_lane = lane_id();
    const int side = (store_lane / 16) % 2;
    const auto u_gc = make_layout<8>(
        make_tuple(number<T::E_M>{}, number<T::E_N / 2>{}, number<8>{}),
        make_tuple(T::T_M * 16 * args.stride_c, number<2 * T::T_N * 16>{}, number<1>{}),
        make_tuple(underscore{}, underscore{}, underscore{}))
        + (wave_m * 16 + store_lane % 16) * args.stride_c
        + (side * T::T_N + wave_n) * 16 + (store_lane / 32) * 8;
    static_assert(layout_load_traits<decltype(u_gc), 8>::r_elem.value == T::E_M * T::E_N / 2);
    const auto co = layout_to_offsets<8>(u_gc);
    auto write_half = [&](const auto& accumulators, int half_m, int half_n) {
        static_for<T::E_M * T::E_N / 2>([&](auto pi) {
            constexpr int ci = decltype(pi)::value * 2;
            const auto first = __builtin_bit_cast(vector_t<unsigned, 2>, cast<D_C>(accumulators[ci]));
            const auto second = __builtin_bit_cast(vector_t<unsigned, 2>, cast<D_C>(accumulators[ci + 1]));
            const auto p0 = __builtin_amdgcn_permlane16_swap(first[0], second[0], false, true);
            const auto p1 = __builtin_amdgcn_permlane16_swap(first[1], second[1], false, true);
            const vector_t<unsigned, 4> packed{p0[0], p1[0], p0[1], p1[1]};
            store<8>(gc, __builtin_bit_cast(vector_t<D_C, 8>, packed),
                co[decltype(pi)::value] + half_m * T::HALF_B_M * args.stride_c
                    + half_n * T::HALF_B_N, 0, number<2>{});
        });
    };
    write_half(v_c[0][0], 0, 0);
    write_half(v_c[1][0], 1, 0);
    write_half(v_c[0][1], 0, 1);
    write_half(v_c[1][1], 1, 1);
    s_waitcnt_vmcnt(number<0>{});

}
#endif // gfx950 device
