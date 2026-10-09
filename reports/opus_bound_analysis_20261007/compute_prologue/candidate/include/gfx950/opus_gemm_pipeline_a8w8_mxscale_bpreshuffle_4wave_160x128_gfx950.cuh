#pragma once

#include <opus/hip_minimal.hpp>
#include <opus/opus.hpp>

#include "opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_4wave_128x128_gfx950.cuh"
#include "opus_gemm_traits_a8w8_mxscale_bpreshuffle_4wave_160x128_gfx950.cuh"

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
    namespace layout_9021 = opus_gemm_4wave_128x128_layout;

    const int wave_id = __builtin_amdgcn_readfirstlane(thread_id_x() / T::WARP_SIZE);
    const int lane_id = thread_id_x() % T::WARP_SIZE;
    const int wave_id_m = wave_id % T::T_M;
    const int wave_id_n = wave_id / T::T_M;
    const int block_m = block_id_y();
    const int block_n = block_id_x();
    const int row = block_m * T::B_M;
    const int col = block_n * T::B_N;
    const int batch_id = block_id_z();
    const int loops = kargs.k / T::B_K;

    // Matrix and scale views retain bounded A/C resources for missing M rows.
    auto g_a = make_gmem(reinterpret_cast<const D_A*>(kargs.ptr_a) + batch_id * kargs.stride_a_batch + row * kargs.stride_a,
                        static_cast<unsigned>((kargs.m - row) * kargs.stride_a));
    auto g_b = make_gmem(reinterpret_cast<const D_B*>(kargs.ptr_b) + batch_id * kargs.stride_b_batch + col * kargs.stride_b);
    auto g_c = make_gmem(reinterpret_cast<D_C*>(kargs.ptr_c) + batch_id * kargs.stride_c_batch + row * kargs.stride_c + col,
                        static_cast<unsigned>(((kargs.m - row) * kargs.stride_c - col) * sizeof(D_C)));
    auto g_sfa = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfa) + batch_id * kargs.stride_sfa_batch + row);
    auto g_sfb = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfb) + batch_id * kargs.stride_sfb_batch + (col / T::GROUP_N) * kargs.stride_sfb);

    // Matrix and scale layouts: global -> LDS -> registers.
    const auto u_ga = make_layout_ga_scale<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_a);
    const auto u_sa = make_layout_sa_scale<T>(wave_id_m, wave_id_n);
    const auto u_ra = make_layout_ra_scale<T>(lane_id, wave_id_m);
    const auto u_gb = make_layout_gb_scale<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_b);
    const auto u_sb = make_layout_sb_scale<T>(wave_id_m, wave_id_n);
    const auto u_rb = make_layout_rb_scale<T>(lane_id, wave_id_n);
    const auto u_gsfa_0 = layout_9021::make_layout_gsfa_scale<T, 0>(lane_id, wave_id_m, wave_id_n, kargs.stride_sfa);
    const auto u_ssfa_0 = layout_9021::make_layout_ssfa_scale<T, 0>(lane_id, wave_id_m, wave_id_n);
    const auto u_gsfa_1 = layout_9021::make_layout_gsfa_scale<T, 1>(lane_id, wave_id_m, wave_id_n, kargs.stride_sfa);
    const auto u_ssfa_1 = layout_9021::make_layout_ssfa_scale<T, 1>(lane_id, wave_id_m, wave_id_n);
    const auto u_rsfa = opus_gemm_8wave_192x256_layout::make_layout_rsfa_scale<T>(lane_id, wave_id_m);
    const auto u_gsfb = layout_9021::make_layout_gsfb_scale<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_sfb);
    const auto u_ssfb = layout_9021::make_layout_ssfb_scale<T>(lane_id, wave_id_m, wave_id_n);
    const auto u_rsfb = layout_9021::make_layout_rsfb_scale<T>();

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
    constexpr int ELEM_A = decltype(mma)::elem_a;
    constexpr int ELEM_B = decltype(mma)::elem_b;
    constexpr int ELEM_C = decltype(mma)::elem_c;

    typename decltype(mma)::vtype_a v_a;
    typename decltype(mma)::vtype_b v_b;
    typename decltype(mma)::vtype_c v_c;
    clear(v_c);

    // C output layout; BF16 fragments are first staged in LDS.
    const auto p_coord_c = opus::make_tuple(wave_id_m, lane_id % mma.grpn_c, wave_id_n, lane_id / mma.grpn_c);
    const auto u_gc = partition_layout_c<T::VEC_C>(mma, opus::make_tuple(number<T::C_LDS_ROW_STRIDE_ELEMS>{}, 1_I), p_coord_c);
    const auto gc_offsets = layout_to_offsets<T::VEC_C>(u_gc);
    array<D_SF_PACK, T::A_SCALE_PACKS> v_sfa{}, v_sfa_next{};
    array<D_SF_PACK, T::B_SCALE_PACKS> v_sfb{}, v_sfb_next{};

    // Matrix, scale panel and C staging offsets.
    auto ga_offset = [&](int tile_k) { return tile_k * number<T::B_K>{}; };
    auto gb_offset = [&](int tile_k) { return tile_k * number<T::B_K * T::W_N>{}; };
    auto sa_offset = [&](int stage) { return stage * T::A_STAGE; };
    auto sb_offset = [&](int stage) { return stage * T::B_STAGE; };
    auto gsfa_offset = [&](int panel_k_begin) { return panel_k_begin * kargs.stride_sfa; };
    auto gsfb_offset = [&](int panel_k_begin) { return panel_k_begin; };
    auto ssfa_offset = [&](int tile_k) { return (tile_k & (T::SCALE_PANEL - 1)) * number<T::B_M>{}; };
    auto ssfb_offset = [&](int tile_k) { return tile_k & (T::SCALE_PANEL - 1); };
    auto c_offset = [&](int row_c, int col_c) { return row_c * T::C_LDS_ROW_STRIDE_ELEMS + col_c; };

    // A/B global memory -> LDS ring stage.
    auto issue_matrix_prefetch = [&](int stage, int tile_k) {
        async_load<T::VEC_A>(g_a, s_a.ptr, u_ga, u_sa + sa_offset(stage), ga_offset(tile_k));
        async_load<T::VEC_B>(g_b, s_b.ptr, u_gb, u_sb + sb_offset(stage), gb_offset(tile_k));
    };
    const auto sfa_gmem_offsets = opus::make_tuple(layout_to_offsets<T::VEC_SCALE_A>(u_gsfa_0), layout_to_offsets<T::VEC_SCALE_A>(u_gsfa_1));
    const auto sfa_smem_offsets = opus::make_tuple(layout_to_offsets<T::VEC_SCALE_A>(u_ssfa_0), layout_to_offsets<T::VEC_SCALE_A>(u_ssfa_1));
    const auto sfb_gmem_offsets = layout_to_offsets<1>(u_gsfb);
    const auto sfb_smem_offsets = layout_to_offsets<1>(u_ssfb);
    const auto rsfa_offsets = layout_to_offsets<1>(u_rsfa);
    const auto rsfb_offsets = layout_to_offsets<1>(u_rsfb);

    // Scale A global memory -> VGPR -> LDS, retaining raw E8M0 bytes.
    auto load_sfa_panel = [&](int panel_k_begin) {
        static_for<T::SFA_PASSES>([&](auto pass_i) {
            constexpr int pass = decltype(pass_i)::value;
            const int smem_offset = opus::get<pass>(sfa_smem_offsets)[0];
            const int local_k_group = smem_offset / T::B_M;
            const int local_row = smem_offset % T::B_M;
            if (smem_offset < T::SFA_BYTES && panel_k_begin + local_k_group < loops) {
                vector_t<D_SF, T::VEC_SCALE_A> raw;
                if (row + local_row < kargs.m)
                    raw = load<T::VEC_SCALE_A>(g_sfa, opus::get<pass>(sfa_gmem_offsets)[0] + gsfa_offset(panel_k_begin));
                else
                    static_for<T::VEC_SCALE_A>([&](auto byte_i) { raw[decltype(byte_i)::value] = 0x7f; });
                store<T::VEC_SCALE_A>(s_sfa, raw, smem_offset);
            }
        });
    };
    // Scale B global memory -> VGPR -> LDS, one byte per K128 group.
    auto load_sfb_panel = [&](int panel_k_begin) {
        const int local_k_group = sfb_smem_offsets[0];
        if (local_k_group < T::SCALE_PANEL && panel_k_begin + local_k_group < loops) {
            store<1>(s_sfb, load<1>(g_sfb, sfb_gmem_offsets[0] + gsfb_offset(panel_k_begin)),
                     sfb_smem_offsets[0]);
        }
    };
    // Scale LDS -> VGPR; pack A repeats after reading LDS.
    auto read_scales = [&](int tile_k, auto& scale_a, auto& scale_b) {
        static_for<T::A_SCALE_PACKS>([&](auto pack_i) { scale_a[decltype(pack_i)::value] = 0; });
        static_for<T::E_M>([&](auto m_i) {
            constexpr int m_repeat = decltype(m_i)::value;
            const D_SF_PACK value = load<1>(s_sfa, rsfa_offsets[m_repeat] + ssfa_offset(tile_k))[0];
            scale_a[m_repeat / 4] |= value << ((m_repeat % 4) * 8);
        });
        static_for<T::B_SCALE_PACKS>([&](auto pack_i) {
            constexpr int pack = decltype(pack_i)::value;
            scale_b[pack] = load<1>(s_sfb, rsfb_offsets[pack] + ssfb_offset(tile_k))[0];
        });
    };
    // Matrix LDS -> registers, one MFMA operand fragment at a time.
    auto load_a_fragment = [&](auto m_i, int stage) {
        layout_9021::load_matrix_fragment<T::VEC_A, ELEM_A>(v_a, s_a, u_ra, m_i, sa_offset(stage));
    };
    auto load_b_fragment = [&](auto n_i, int stage) {
        layout_9021::load_matrix_fragment<T::VEC_B, ELEM_B>(v_b, s_b, u_rb, n_i, sb_offset(stage));
    };
    auto mma_scale_fragment = [&](auto m_i, auto n_i) {
        constexpr int m_repeat = decltype(m_i)::value;
        constexpr int n_repeat = decltype(n_i)::value;
        constexpr int c_index = m_repeat * T::E_N + n_repeat;
        constexpr int scale_n_index = n_repeat / (T::GROUP_N / (T::T_N * T::W_N));
        const auto a = slice(v_a, number<m_repeat * ELEM_A>{}, number<(m_repeat + 1) * ELEM_A>{});
        const auto b = slice(v_b, number<n_repeat * ELEM_B>{}, number<(n_repeat + 1) * ELEM_B>{});
        auto c = slice(v_c, number<c_index * ELEM_C>{}, number<(c_index + 1) * ELEM_C>{});
        c = typename decltype(mma)::MMA{}(a, b, c,
                                        static_cast<int>(v_sfa[m_repeat / 4]), static_cast<int>(v_sfb[scale_n_index]),
                                        number<m_repeat % 4>{}, number<0>{});
        set_slice(v_c, c, number<c_index * ELEM_C>{}, number<(c_index + 1) * ELEM_C>{});
    };

    int stage = 0;

    // Prologue
    issue_matrix_prefetch(0, 0);
    __builtin_amdgcn_sched_barrier(0);
    load_sfa_panel(0);
    load_sfb_panel(0);
    __builtin_amdgcn_sched_barrier(0);
    if (loops > 1) {
        issue_matrix_prefetch(1, 1);
        __builtin_amdgcn_sched_barrier(0);
        s_waitcnt_vmcnt(number<T::VMEM_INSTRUCTIONS_PER_TILE>{});
    } else {
        s_waitcnt_vmcnt(0_I);
    }
    s_waitcnt_lgkmcnt(0_I);
    __builtin_amdgcn_s_barrier();
    read_scales(0, v_sfa, v_sfb);
    static_for<T::E_M>([&](auto m_i) { load_a_fragment(m_i, 0); });
    static_for<T::E_N>([&](auto n_i) { load_b_fragment(n_i, 0); });
    s_waitcnt_lgkmcnt(0_I);
    if constexpr (T::NUM_STAGES == 2)
        __builtin_amdgcn_s_barrier();

    // Main loop
    for (int tile_k = 0; tile_k + 1 < loops; ++tile_k) {
        const int next_stage = stage + 1 == T::NUM_STAGES ? 0 : stage + 1;
        const int future_stage = T::NUM_STAGES == 2 ? stage : (stage == 0 ? 2 : stage - 1);
        const bool has_future = tile_k + 2 < loops;
        if (has_future)
            issue_matrix_prefetch(future_stage, tile_k + 2);

        if (((tile_k + 1) & (T::SCALE_PANEL - 1)) == 0) {
            load_sfa_panel(tile_k + 1);
            load_sfb_panel(tile_k + 1);
            s_waitcnt_vmcnt(0_I);
            s_waitcnt_lgkmcnt(0_I);
            __builtin_amdgcn_s_barrier();
        }
        read_scales(tile_k + 1, v_sfa_next, v_sfb_next);

        static_for<T::E_M>([&](auto m_i) {
            static_for<T::E_N>([&](auto n_i) {
                mma_scale_fragment(m_i, n_i);
                if constexpr (decltype(m_i)::value == T::E_M - 1)
                    load_b_fragment(n_i, next_stage);
            });
            if constexpr (decltype(m_i)::value == 0) {

                __builtin_amdgcn_sched_barrier(0);
                if (has_future)
                    s_waitcnt_vmcnt(number<T::VMEM_INSTRUCTIONS_PER_TILE>{});
                else
                    s_waitcnt_vmcnt(0_I);
                s_waitcnt_lgkmcnt(0_I);
                __builtin_amdgcn_s_barrier();
                __builtin_amdgcn_sched_barrier(0);
            }
            load_a_fragment(m_i, next_stage);
        });

        if constexpr (T::NUM_STAGES == 2) {
            s_waitcnt_lgkmcnt(0_I);
            __builtin_amdgcn_s_barrier();
        }
        v_sfa = v_sfa_next;
        v_sfb = v_sfb_next;
        stage = stage + 1 == T::NUM_STAGES ? 0 : stage + 1;
    }

    // Epilogue
    static_for<T::E_M>([&](auto m_i) {
        static_for<T::E_N>([&](auto n_i) { mma_scale_fragment(m_i, n_i); });
    });

    if constexpr (T::NUM_STAGES == 3) {
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_barrier();
    }

    // Accumulator -> BF16 -> LDS.
    auto stage_output_fragment = [&](auto c_i) {
        constexpr int c_index = decltype(c_i)::value;
        const auto c = slice(v_c, number<c_index * ELEM_C>{}, number<(c_index + 1) * ELEM_C>{});
        store<T::VEC_C>(s_c, cast<D_C>(c), gc_offsets[c_index]);
    };

    const int output_thread_id = wave_id * T::WARP_SIZE + lane_id;
    auto copy_output_bf16 = [&](auto copy_i) {
        const int linear = output_thread_id * T::VEC_OUTPUT + decltype(copy_i)::value * T::BLOCK_SIZE * T::VEC_OUTPUT;
        const int output_row = linear / T::B_N;
        const int output_col = linear % T::B_N;
        const auto value = load<T::VEC_OUTPUT>(s_c, c_offset(output_row, output_col));
        store<T::VEC_OUTPUT>(g_c, value, output_row * kargs.stride_c + output_col, 0, opus::number<2>{});
    };

    static_for<T::E_M * T::E_N>([&](auto c_i) { stage_output_fragment(c_i); });
    s_waitcnt_lgkmcnt(0_I);
    __builtin_amdgcn_s_barrier();
    static_for<T::OUTPUT_PASSES>([&](auto copy_i) { copy_output_bf16(copy_i); });
}
#endif
