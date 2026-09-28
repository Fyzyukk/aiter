// SPDX-License-Identifier: Apache-2.0
// Runtime K128..16384, 160x128, four Wave64, native E8M0 and M64 tails.
// Reuses unchanged Apache-2.0 matrix layout helpers from csrc/opus_gemm/include/gfx950.
#pragma once
#include <opus/hip_minimal.hpp>
#include <opus/opus.hpp>

#include "opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_4wave_gfx950.cuh"
#include "traits_runtime.cuh"

#if !defined(__HIP_DEVICE_COMPILE__)
template<class Traits>
__global__ void gemm_a8w8_mxfp8_scale_4wave_160x128_kernel(opus_gemm_mxscale_bpreshuffle_kargs_gfx950) {}
#elif defined(__gfx950__)
template<class Traits>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, Traits::MIN_WGS_PER_CU)
void gemm_a8w8_mxfp8_scale_4wave_160x128_kernel(opus_gemm_mxscale_bpreshuffle_kargs_gfx950 kargs) {
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
    const int block_m = block_id_y();
    const int block_n = block_id_x();
    const int row = block_m * T::B_M;
    const int col = block_n * T::B_N;

    const int loops = kargs.k / T::B_K;

    // Matrix global-memory views.
    // Dispatch guarantees M%64=N%128=K%128=0, 128<=K<=16384 and int32 extents.
    // Buffer OOB zero-fills missing A rows and discards missing output rows.
    auto g_a = make_gmem(reinterpret_cast<const D_A*>(kargs.ptr_a) + row * kargs.stride_a,
                        static_cast<unsigned>((kargs.m - row) * kargs.stride_a));
    auto g_b = make_gmem(reinterpret_cast<const D_B*>(kargs.ptr_b) + col * kargs.stride_b);
    auto g_c = make_gmem(reinterpret_cast<D_C*>(kargs.ptr_c) + row * kargs.stride_c + col,
                        static_cast<unsigned>(((kargs.m - row) * kargs.stride_c - col) * sizeof(D_C)));
    auto g_sfa = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfa));
    auto g_sfb = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfb));

    // Matrix layouts: global -> LDS -> registers.
    const auto u_ga = make_layout_ga_scale<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_a);
    const auto u_sa = make_layout_sa_scale<T>(wave_id_m, wave_id_n);
    const auto u_gb = make_layout_gb_scale<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_b);
    const auto u_sb = make_layout_sb_scale<T>(wave_id_m, wave_id_n);
    const auto u_rb = make_layout_rb_scale<T>(lane_id, wave_id_n);

    // Matrix and scale LDS; matrix storage is reused for the C epilogue.
    constexpr int c_lds_row_stride_elems = T::B_N + 8;
    constexpr int c_lds_bytes = T::B_M * c_lds_row_stride_elems * sizeof(D_C);
    constexpr int OUTPUT_VEC = 8;
    constexpr int OUTPUT_PASSES = T::B_M * T::B_N / (T::BLOCK_SIZE * OUTPUT_VEC);
    static_assert(c_lds_row_stride_elems == 136 && c_lds_bytes <= T::LDS_BYTES);
    static_assert(T::B_M * T::B_N % (T::BLOCK_SIZE * OUTPUT_VEC) == 0);
    alignas(16) __shared__ char smem_matrix[T::LDS_BYTES];
    auto s_a = make_smem(reinterpret_cast<D_A*>(smem_matrix));
    auto s_b = make_smem(reinterpret_cast<D_B*>(smem_matrix + T::NUM_STAGES * T::A_STAGE));
    auto s_c = make_smem(reinterpret_cast<D_C*>(smem_matrix));
    auto s_sfa = make_smem(reinterpret_cast<D_SF*>(smem_matrix + T::MATRIX_LDS_BYTES));
    auto s_sfb = make_smem(reinterpret_cast<D_SF*>(smem_matrix + T::MATRIX_LDS_BYTES + T::SFA_BYTES));

    // MMA and register fragments.
    auto mma = make_tiled_mma<D_A, D_B, D_ACC>(
        seq<T::E_M, T::E_N, 1>{}, seq<T::T_M, T::T_N, 1>{},
        seq<16, 16, 128>{}, mfma_adaptor_swap_ab{});
    using BaseMMA = typename decltype(mma)::MMA;
    using AFragment = typename BaseMMA::vtype_a;
    using BFragment = typename BaseMMA::vtype_b;
    using AccFragment = typename BaseMMA::vtype_c;
    using AChunk = vector_t<D_A, 16>;
    using BChunk = vector_t<D_B, 16>;
    array<AFragment, T::E_M> v_a;
    array<BFragment, T::E_N> v_b;
    array<AccFragment, T::E_M * T::E_N> c00{};
    array<D_SF_PACK, T::A_SCALE_PACKS> v_sfa{}, v_sfa_next{};
    array<D_SF_PACK, T::B_SCALE_PACKS> v_sfb{}, v_sfb_next{};
    auto* a_chunks = reinterpret_cast<AChunk*>(&v_a);
    auto* b_chunks = reinterpret_cast<BChunk*>(&v_b);

    // Matrix offsets for global K tiles and LDS ring stages.
    auto ga_offset = [&](int tile) { return tile * T::B_K; };
    auto gb_offset = [&](int tile) { return tile * T::B_K * T::W_N; };
    auto sa_offset = [&](int stage) { return stage * T::A_STAGE; };
    auto sb_offset = [&](int stage) { return stage * T::B_STAGE; };

    // A/B K-tile requests copy global memory directly into an LDS ring slot.
    auto issue_matrix_prefetch = [&](int stage, int tile) {
        async_load<16>(g_a, s_a.ptr, u_ga, u_sa + sa_offset(stage), ga_offset(tile));
        async_load<16>(g_b, s_b.ptr, u_gb, u_sb + sb_offset(stage), gb_offset(tile));
    };

    // Scale-panel publication and packed register reads.
    auto load_scale_panel = [&](int panel_begin) {
        static_for<T::SFA_PASSES>([&](auto i) {
            const int index = (thread_id_x() + decltype(i)::value * T::BLOCK_SIZE) * 16;
            const int group = index / T::B_M;
            const int local_row = index % T::B_M;
            if (index < T::SFA_BYTES && panel_begin + group < loops) {
                vector_t<D_SF, 16> values;
                if (row + local_row < kargs.m)
                    values = load<16>(g_sfa, (panel_begin + group) * kargs.stride_sfa + row + local_row);
                else
                    static_for<16>([&](auto j) { values[decltype(j)::value] = 0x7f; });
                store<16>(s_sfa, values, index);
            }
        });
        // BN128 has exactly one B-scale row. Do not load unused K groups.
        if (thread_id_x() < T::SCALE_PANEL && panel_begin + thread_id_x() < loops) {
            const int group = thread_id_x();
            store<1>(s_sfb, load<1>(g_sfb, (col / T::GROUP_N) * kargs.stride_sfb + panel_begin + group),
                     thread_id_x());
        }
    };
    auto read_scales = [&](int group, auto& scale_a, auto& scale_b) {
        const int panel_group = group & (T::SCALE_PANEL - 1);
        static_for<T::A_SCALE_PACKS>([&](auto p) { scale_a[decltype(p)::value] = 0; });
        static_for<T::E_M>([&](auto i) {
            constexpr int repeat = decltype(i)::value;
            const int local_row = wave_id_m * T::W_M + lane_id % T::W_M + repeat * T::T_M * T::W_M;
            const D_SF_PACK value = load<1>(s_sfa, panel_group * T::B_M + local_row)[0];
            scale_a[repeat / 4] |= value << ((repeat % 4) * 8);
        });
        static_for<T::B_SCALE_PACKS>([&](auto i) {
            constexpr int half_n = decltype(i)::value;
            scale_b[half_n] = load<1>(s_sfb, half_n * T::SCALE_PANEL + panel_group)[0];
        });
    };
    // LDS operand reads and one accumulator update.
    auto load_a = [&](auto m, int stage) {
        constexpr int repeat = decltype(m)::value;
        static_for<2>([&](auto c) {
            constexpr int index = repeat * 2 + decltype(c)::value;
            const int offset = T::a_lds_offset(wave_id_m, lane_id, repeat, decltype(c)::value) + sa_offset(stage);
            // C AGPR0:79, A80:119.
            [[clang::amdgpu_pin_agpr(T::C_REGS + index * 4)]]
            a_chunks[index] = load<16>(s_a, offset);
        });
    };
    auto load_b = [&](auto n, int stage) {
        constexpr int repeat = decltype(n)::value;
        const auto offsets = layout_to_offsets<16>(u_rb + sb_offset(stage));
        static_for<2>([&](auto c) {
            constexpr int index = repeat * 2 + decltype(c)::value;
            b_chunks[index] = load<16>(s_b, offsets[index]);
        });
    };
    auto mma_scale_fragment = [&](auto m, auto n) {
        constexpr int mi = decltype(m)::value, ni = decltype(n)::value;
        constexpr int ci = mi * T::E_N + ni;
        constexpr int half_n = ni / (T::GROUP_N / (T::T_N * T::W_N));
        [[clang::amdgpu_pin_agpr(ci * sizeof(AccFragment) / sizeof(u32_t))]]
        c00[ci] = BaseMMA{}(v_a[mi], v_b[ni], c00[ci],
                           static_cast<int>(v_sfa[mi / 4]), static_cast<int>(v_sfb[half_n]),
                           number<mi % 4>{}, number<0>{});
    };

    int stage = 0;

    // ===== Prologue =====
    // Seed only existing groups. One-tile K128 never loads or reads K1.
    load_scale_panel(0);
    issue_matrix_prefetch(0, 0);
    if (loops > 1)
        issue_matrix_prefetch(1, 1);
    s_waitcnt_vmcnt(0_I);
    s_waitcnt_lgkmcnt(0_I);
    __builtin_amdgcn_s_barrier();
    read_scales(0, v_sfa, v_sfb);
    static_for<T::E_M>([&](auto m) { load_a(m, 0); });
    static_for<T::E_N>([&](auto n) { load_b(n, 0); });
    s_waitcnt_lgkmcnt(0_I);
    // The two-stage ring reuses K0 immediately for K2, so retire its readers.
    if constexpr (T::NUM_STAGES == 2)
        __builtin_amdgcn_s_barrier();

    // ===== Main loop =====
    // One runtime U1 body owns the matrix ring and drain. Track the scalar
    // stage with one increment/wrap; no per-stage or fixed-K loop copies.
#pragma clang loop unroll(disable)
    for (int tile = 0; tile + 1 < loops; ++tile) {
        const int next_stage = stage + 1 == T::NUM_STAGES ? 0 : stage + 1;
        const int future_stage = T::NUM_STAGES == 2 ? stage : (stage == 0 ? 2 : stage - 1);
        // Prefetch K+2 into its ring slot. The two-stage count is fixed by
        // this tile geometry; runtime K never selects a different flow.
        const bool has_future = tile + 2 < loops;
        if (has_future)
            issue_matrix_prefetch(future_stage, tile + 2);
        // Current scales were read before the preceding publication barrier,
        // so all waves have retired those panel reads before ring reuse.
        // The current tile's scales are in registers while the panel refills.
        if (((tile + 1) & (T::SCALE_PANEL - 1)) == 0) {
            load_scale_panel(tile + 1);
            s_waitcnt_vmcnt(0_I);
            s_waitcnt_lgkmcnt(0_I);
            __builtin_amdgcn_s_barrier();
        }
        read_scales(tile + 1, v_sfa_next, v_sfb_next);
        // Preserve the original last-use replacement of each operand.
        static_for<T::E_M>([&](auto m) {
            static_for<T::E_N>([&](auto n) {
                mma_scale_fragment(m, n);
                if constexpr (decltype(m)::value == T::E_M - 1)
                    load_b(n, next_stage);
            });
            if constexpr (decltype(m)::value == 0) {
                // The first M repeat consumes only current register operands.
                // Before the first next-tile LDS read, retire the older K+1
                // matrix requests while permitting this step's K+2 requests
                // to stay in flight. The no-future drain must wait for all.
                __builtin_amdgcn_sched_barrier(0);
                if (has_future)
                    s_waitcnt_vmcnt(number<T::VMEM_INSTRUCTIONS_PER_TILE>{});
                else
                    s_waitcnt_vmcnt(0_I);
                s_waitcnt_lgkmcnt(0_I);
                __builtin_amdgcn_s_barrier();
                __builtin_amdgcn_sched_barrier(0);
            }
            load_a(m, next_stage);
        });
        // The next advance immediately reuses the just-read next slot for
        // K+3 and needs a consumer barrier.
        if constexpr (T::NUM_STAGES == 2) {
            s_waitcnt_lgkmcnt(0_I);
            __builtin_amdgcn_s_barrier();
        }
        v_sfa = v_sfa_next;
        v_sfb = v_sfb_next;
        stage = stage + 1 == T::NUM_STAGES ? 0 : stage + 1;
    }

    // ===== Epilogue =====
    static_for<T::E_M>([&](auto m) {
        static_for<T::E_N>([&](auto n) { mma_scale_fragment(m, n); });
    });

    // The seed or final no-future advance drained all matrix VMEM requests.
    // The two-stage ring already retired the last operand readers at
    // its final end barrier before LDS becomes output.
    if constexpr (T::NUM_STAGES == 3) {
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_barrier();
    }

    // ===== Output writeback =====
    const auto p_coord_c = opus::make_tuple(wave_id_m, lane_id % mma.grpn_c, wave_id_n, lane_id / mma.grpn_c);
    const auto u_gc = partition_layout_c<T::VEC_C>(mma, opus::make_tuple(c_lds_row_stride_elems, 1_I), p_coord_c);
    const auto gc_offsets = layout_to_offsets<T::VEC_C>(u_gc);

    // AGPR -> BF16 -> LDS.
    auto stage_output_fragment = [&](auto i) {
        constexpr int index = decltype(i)::value;
        store<T::VEC_C>(s_c, cast<D_C>(c00[index]), gc_offsets[index]);
    };

    // Recombine MFMA rows into contiguous 16-byte global stores.
    auto copy_output_bf16 = [&](auto pass) {
        const int linear = thread_id_x() * OUTPUT_VEC + decltype(pass)::value * T::BLOCK_SIZE * OUTPUT_VEC;
        const int output_row = linear / T::B_N;
        const int output_col = linear % T::B_N;
        const auto value = load<OUTPUT_VEC>(s_c, output_row * c_lds_row_stride_elems + output_col);
        store<OUTPUT_VEC>(g_c, value, output_row * kargs.stride_c + output_col, 0, opus::number<2>{});
    };

    static_for<T::E_M * T::E_N>([&](auto i) { stage_output_fragment(i); });
    s_waitcnt_lgkmcnt(0_I);
    __builtin_amdgcn_s_barrier();
    static_for<OUTPUT_PASSES>([&](auto pass) { copy_output_bf16(pass); });
}
#endif
