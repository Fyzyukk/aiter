// SPDX-License-Identifier: Apache-2.0
// Large-output runtime-K128..16384 kernel, 192x256, eight Wave64, E8M0/M64 tails.
// Unified U2 loop, 128-group resident scales, fused final MFMA/BF16 staging,
// and a shape-based workgroup mapping; every K follows this one device flow.
// Uses 64-bit C base addresses and bounded tile-local output ranges.
// Reuses unchanged Apache-2.0 helpers from csrc/opus_gemm/include/.
#pragma once

#include <opus/hip_minimal.hpp>
#include <opus/opus.hpp>

#include <cstdint>

#include "opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_4wave_gfx950.cuh"
#include "opus_gemm_traits_a8w8_mxscale_bpreshuffle_8wave_192x256_large_output_gfx950.cuh"

#if !defined(__HIP_DEVICE_COMPILE__)
template<class Traits>
__global__ void gemm_a8w8_mxfp8_scale_8wave_192x256_large_output_kernel(opus_gemm_mxscale_bpreshuffle_kargs_gfx950) {}
#elif defined(__gfx950__)
template<class Traits>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, Traits::MIN_WGS_PER_CU)
void gemm_a8w8_mxfp8_scale_8wave_192x256_large_output_kernel(opus_gemm_mxscale_bpreshuffle_kargs_gfx950 kargs) {
    using namespace opus;

    using T = opus::remove_cvref_t<Traits>;
    using D_A = opus::fp8_t;
    using D_B = opus::fp8_t;
    using D_C = opus::bf16_t;
    using D_ACC = opus::fp32_t;
    using D_SF = unsigned char;
    using D_SF_PACK = unsigned int;

    // Tile and thread coordinates.
    // Keep wave-uniform values before tile traversal, preserving their lowering.
    const int wave_id = __builtin_amdgcn_readfirstlane(thread_id_x() / T::WARP_SIZE);
    const int lane_id = thread_id_x() % T::WARP_SIZE;
    const int wave_id_m = wave_id % T::T_M;
    const int wave_id_n = wave_id / T::T_M;
    // One workgroup-uniform policy for every runtime K. Group four M tiles
    // on tall/narrow matrices to reuse B; retain N-first elsewhere.
    int block_m = block_id_y();
    int block_n = block_id_x();
    if (kargs.n <= 2048 && kargs.m >= 4096) {
        constexpr int GROUP_M = 4;
        const int grid_m = 1 + (kargs.m - 1) / T::B_M;
        const int grid_n = kargs.n / T::B_N;
        const int linear = block_id_y() * grid_n + block_id_x();
        const int group_size = GROUP_M * grid_n;
        const int first_m = (linear / group_size) * GROUP_M;
        const int remaining_m = grid_m - first_m;
        const int actual_m = remaining_m < GROUP_M ? remaining_m : GROUP_M;
        const int within_group = linear % group_size;
        // The final M group may be shorter; map every tile exactly once.
        block_m = first_m + within_group % actual_m;
        block_n = within_group / actual_m;
    }
    const int row = block_m * T::B_M;
    const int col = block_n * T::B_N;

    const int loops = kargs.k / T::B_K;

    // Matrix global-memory views.
    // Dispatch guarantees M%64=N%256=K%128=0 and 128<=K<=16384.
    // A/B extents and the C tile byte span fit signed 32-bit addressing.
    // Buffer OOB zero-fills missing A rows and discards missing output rows.
    auto g_a = make_gmem(reinterpret_cast<const D_A*>(kargs.ptr_a) + row * kargs.stride_a,
                        static_cast<unsigned>((kargs.m - row) * kargs.stride_a));
    auto g_b = make_gmem(reinterpret_cast<const D_B*>(kargs.ptr_b) + col * kargs.stride_b);
    const int remaining_rows = kargs.m - row;
    const int valid_rows = remaining_rows < T::B_M ? remaining_rows : T::B_M;
    const int64_t c_offset = static_cast<int64_t>(row) * kargs.stride_c + col;
    const unsigned c_bytes = static_cast<unsigned>(
        ((static_cast<int64_t>(valid_rows) - 1) * kargs.stride_c + T::B_N) * sizeof(D_C));
    auto g_c = make_gmem(reinterpret_cast<D_C*>(kargs.ptr_c) + c_offset, c_bytes);
    auto g_sfa = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfa));
    auto g_sfb = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfb));

    // Matrix layouts: global -> LDS -> registers.
    const auto u_ga = make_layout_ga_scale<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_a);
    const auto u_sa = make_layout_sa_scale<T>(wave_id_m, wave_id_n);
    const auto u_gb = make_layout_gb_scale<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_b);
    const auto u_sb = make_layout_sb_scale<T>(wave_id_m, wave_id_n);
    const auto u_rb = make_layout_rb_scale<T>(lane_id, wave_id_n);

    // Matrix and scale LDS; matrix storage is reused for the C epilogue.
    alignas(16) __shared__ char smem_matrix[T::LDS_BYTES];
    auto s_a = make_smem(reinterpret_cast<D_A*>(smem_matrix));
    auto s_b = make_smem(reinterpret_cast<D_B*>(smem_matrix + T::NUM_STAGES * T::A_STAGE));
    auto s_c = make_smem(reinterpret_cast<D_C*>(smem_matrix));
    auto s_sfa = make_smem(reinterpret_cast<D_SF*>(smem_matrix + T::MATRIX_LDS_BYTES));
    auto s_sfb = make_smem(reinterpret_cast<D_SF*>(smem_matrix + T::MATRIX_LDS_BYTES + T::SFA_BYTES));
    // MMA and register fragments.
    auto mma = make_tiled_mma<D_A, D_B, D_ACC>(
        seq<T::E_M, T::E_N, T::E_K>{},
        seq<T::T_M, T::T_N, T::T_K>{},
        seq<T::W_M, T::W_N, T::W_K>{},
        mfma_adaptor_swap_ab{});
    using BaseMMA = typename decltype(mma)::MMA;
    using AFragment = typename BaseMMA::vtype_a;
    using BFragment = typename BaseMMA::vtype_b;
    using AccFragment = typename BaseMMA::vtype_c;
    using AChunk = opus::vector_t<D_A, T::VEC_A>;
    using BChunk = opus::vector_t<D_B, T::VEC_B>;
    using CTile = opus::array<AccFragment, T::E_M * T::E_N>;
    array<AFragment, T::E_M> v_a;
    array<BFragment, T::E_N> v_b;
    CTile c00{};
    array<D_SF_PACK, T::A_SCALE_PACKS> v_sfa{}, v_sfa_next{};
    array<D_SF_PACK, T::B_SCALE_PACKS> v_sfb{}, v_sfb_next{};
    auto* a_chunks = reinterpret_cast<AChunk*>(&v_a);
    auto* b_chunks = reinterpret_cast<BChunk*>(&v_b);

    // C row stride in D_C elements while staging BF16 output in LDS.
    constexpr int c_lds_row_stride_elems = T::B_N + 8;
    constexpr int OUTPUT_PASSES = T::B_M * T::B_N / (T::BLOCK_SIZE * 8);
    static_assert(T::B_M * c_lds_row_stride_elems * sizeof(D_C) <= T::LDS_BYTES);
    static_assert(T::B_M * T::B_N % (T::BLOCK_SIZE * 8) == 0);

    // Matrix offsets and pipeline helpers.
    auto ga_offset = [&](int tile) { return tile * number<T::B_K>{}; };
    auto gb_offset = [&](int tile) { return tile * number<T::B_K * T::W_N>{}; };
    auto sa_offset = [&](int stage) { return stage * T::A_STAGE; };
    auto sb_offset = [&](int stage) { return stage * T::B_STAGE; };

    // Map an A fragment to the cooperative eight-wave LDS layout.
    auto a_lds_offset = [](int wave_m, int lane, int repeat, int chunk) {
        const int matrix_row = repeat * T::T_M * T::W_M + wave_m * T::W_M + lane % T::W_M;
        constexpr int rows_per_pass = T::BLOCK_SIZE / (T::B_K / T::VEC_A);
        const int pass = matrix_row / rows_per_pass;
        const int remainder = matrix_row % rows_per_pass;
        const int producer_n = remainder / (8 * T::T_M);
        const int producer_m = remainder % T::T_M;
        const int producer_lane_row = (remainder % (8 * T::T_M)) / T::T_M;
        return (pass * T::NUM_WAVES + producer_n * T::T_M + producer_m) *
                   (T::smem_linear_wave + T::smem_padding) +
               producer_lane_row * T::B_K + (lane / T::W_M) * T::VEC_A + chunk * 64;
    };

    auto issue_matrix_prefetch = [&](auto stage, int tile) {
        constexpr int slot = decltype(stage)::value;
        const int group = tile;
        static_assert(slot >= 0 && slot < T::NUM_STAGES);
        async_load<16>(g_a, s_a.ptr, u_ga, u_sa + sa_offset(slot), ga_offset(group));
        async_load<16>(g_b, s_b.ptr, u_gb, u_sb + sb_offset(slot),
                       gb_offset(group));
    };
    auto load_scale_panel = [&]() {
        static_for<T::SFA_PASSES>([&](auto i) {
            const int index = (thread_id_x() + decltype(i)::value * T::BLOCK_SIZE) * 16;
            const int group = index / T::B_M;
            const int local_row = index % T::B_M;
            if (index < T::SFA_BYTES && group < loops) {
                vector_t<D_SF, 16> values;
                if (row + local_row < kargs.m)
                    values = load<16>(g_sfa, group * kargs.stride_sfa + row + local_row);
                else
                    static_for<16>([&](auto j) { values[decltype(j)::value] = 0x7f; });
                store<16>(s_sfa, values, index);
            }
        });
        if (thread_id_x() < T::SFB_BYTES) {
            const int half_n = thread_id_x() / T::SCALE_PANEL;
            const int group = thread_id_x() % T::SCALE_PANEL;
            if (group < loops)
                store<1>(s_sfb, load<1>(g_sfb, (col / T::GROUP_N + half_n) * kargs.stride_sfb + group),
                         thread_id_x());
        }
    };
    auto read_scales = [&](int tile, auto& scale_a, auto& scale_b) {
        const int group = tile;
        static_for<T::A_SCALE_PACKS>([&](auto p) { scale_a[decltype(p)::value] = 0; });
        static_for<T::E_M>([&](auto i) {
            constexpr int repeat = decltype(i)::value;
            const int local_row = wave_id_m * T::W_M + lane_id % T::W_M + repeat * T::T_M * T::W_M;
            const unsigned value = load<1>(s_sfa, group * number<T::B_M>{} + local_row)[0];
            scale_a[repeat / 4] |= value << ((repeat % 4) * 8);
        });
        static_for<T::B_SCALE_PACKS>([&](auto i) {
            constexpr int half_n = decltype(i)::value;
            scale_b[half_n] = load<1>(s_sfb, half_n * T::SCALE_PANEL + group)[0];
        });
    };
    auto load_a = [&](auto m, auto stage) {
        constexpr int repeat = decltype(m)::value;
        constexpr int slot = decltype(stage)::value;
        static_for<2>([&](auto c) {
            constexpr int index = repeat * 2 + decltype(c)::value;
            const int offset = a_lds_offset(wave_id_m, lane_id, repeat, decltype(c)::value) + sa_offset(slot);
            a_chunks[index] = load<16>(s_a, offset);
        });
    };
    auto load_b = [&](auto n, auto stage) {
        constexpr int repeat = decltype(n)::value;
        constexpr int slot = decltype(stage)::value;
        const auto offsets = layout_to_offsets<16>(u_rb + sb_offset(slot));
        static_for<2>([&](auto c) {
            constexpr int index = repeat * 2 + decltype(c)::value;
            b_chunks[index] = load<16>(s_b, offsets[index]);
        });
    };
    auto mma_scale_fragment = [&](auto m, auto n) {
        constexpr int mi = decltype(m)::value, ni = decltype(n)::value;
        constexpr int ci = mi * T::E_N + ni;
        constexpr int half_n = ni / (T::GROUP_N / (T::T_N * T::W_N));
        c00[ci] = BaseMMA{}(v_a[mi], v_b[ni], c00[ci],
                           static_cast<int>(v_sfa[mi / 4]), static_cast<int>(v_sfb[half_n]),
                           number<mi % 4>{}, number<0>{});
    };

    auto advance_tile = [&](auto stage, int tile) {
        constexpr int slot = decltype(stage)::value;
        constexpr int next_stage = (slot + 1) % T::NUM_STAGES;
        // Current operands are in registers; K+2 can replace their LDS slot.
        // The stage is static, while the prefetch bound is wave-uniform.
        if (tile + 2 < loops)
            issue_matrix_prefetch(stage, tile + 2);
        read_scales(tile + 1, v_sfa_next, v_sfb_next);
        // Replace each A after its last N use, and B after the last M use.
        static_for<T::E_M>([&](auto m) {
            static_for<T::E_N>([&](auto n) {
                mma_scale_fragment(m, n);
                if constexpr (decltype(m)::value == T::E_M - 1)
                    load_b(n, number<next_stage>{});
            });
            load_a(m, number<next_stage>{});
        });
        // Publish the future slot and finish all reads before slot reuse.
        s_waitcnt_vmcnt(0_I);
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_barrier();
        v_sfa = v_sfa_next;
        v_sfb = v_sfb_next;
    };

    // ===== Prologue =====
    // Load only valid scale groups once. No scale refill path is needed.
    load_scale_panel();
    issue_matrix_prefetch(number<0>{}, 0);
    if (loops > 1) issue_matrix_prefetch(number<1>{}, 1);
    s_waitcnt_vmcnt(0_I);
    s_waitcnt_lgkmcnt(0_I);
    __builtin_amdgcn_s_barrier();
    read_scales(0, v_sfa, v_sfb);
    static_for<T::E_M>([&](auto m) { load_a(m, number<0>{}); });
    static_for<T::E_N>([&](auto n) { load_b(n, number<0>{}); });
    s_waitcnt_lgkmcnt(0_I);
    // Every wave must finish reading K0 before any producer reuses its slot.
    __builtin_amdgcn_s_barrier();

    // ===== Main loop =====
    // A single runtime U2 loop also drains odd/even group counts. There
    // is one compiled advance per static stage, not one kernel per K value.
    // K128 skips the loop; its K0 register fragments go straight to final compute.
#pragma clang loop unroll(disable)
    for (int group = 0; group + 1 < loops; group += 2) {
        advance_tile(number<0>{}, group);
        // For an even group count, stage0 above is the final drain. For an
        // odd group count, stage1 below is the final drain. Both load only
        // the real last group and suppress a nonexistent future prefetch.
        if (group + 2 < loops)
            advance_tile(number<1>{}, group + 1);
    }
    // ===== Epilogue =====
    // K128 uses the second prologue lgkmcnt0/barrier; every larger K
    // ends with an advance whose final A/B/scale reads precede its full
    // waits/barrier. The final compute uses registers only, so there is
    // no repeated pre-output sync. Keep the output publication below.
    const auto p_coord = opus::make_tuple(wave_id_m, lane_id % mma.grpn_c, wave_id_n, lane_id / mma.grpn_c);
    const auto u_sc = partition_layout_c<T::VEC_C>(mma, opus::make_tuple(number<c_lds_row_stride_elems>{}, 1_I), p_coord);
    const auto offsets = layout_to_offsets<T::VEC_C>(u_sc);
    // All final-group operands are in registers. Consume each final MFMA
    // once and stage its completed BF16 fragment while independent MFMAs run.
    static_for<T::E_M>([&](auto m) {
        static_for<T::E_N>([&](auto n) {
            constexpr int index = decltype(m)::value * T::E_N + decltype(n)::value;
            mma_scale_fragment(m, n);
            store<T::VEC_C>(s_c, cast<D_C>(c00[index]), offsets[index]);
        });
    });
    // ===== Output writeback =====
    // Publish the complete BF16 tile before cooperative contiguous reads.
    s_waitcnt_lgkmcnt(0_I);
    __builtin_amdgcn_s_barrier();
    const int output_thread_id = wave_id * T::WARP_SIZE + lane_id;
    auto copy_output_bf16 = [&](auto i) {
        constexpr int pass = decltype(i)::value;
        const int linear = output_thread_id * 8 + pass * T::BLOCK_SIZE * 8;
        const int output_row = linear / T::B_N;
        const int output_col = linear % T::B_N;
        const auto value = load<8>(s_c, output_row * c_lds_row_stride_elems + output_col);
        // g_c has a 64-bit tile base and a bounded tile-local byte range.
        // Check missing rows before forming the signed-int local offset.
        if (row + output_row < kargs.m)
            store<8>(g_c, value, output_row * kargs.stride_c + output_col,
                     0, opus::number<2>{});
    };
    static_for<OUTPUT_PASSES>(copy_output_bf16);
}
#endif
