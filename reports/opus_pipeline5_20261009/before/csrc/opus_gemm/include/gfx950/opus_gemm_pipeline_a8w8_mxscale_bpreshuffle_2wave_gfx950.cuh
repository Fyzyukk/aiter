// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_4wave_128x128_gfx950.cuh"

#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
template<class Traits>
__device__ inline void gemm_a8w8_mxfp8_scale_2wave(opus_gemm_mxscale_bpreshuffle_kargs_gfx950 kargs) {
    using namespace opus;

    using T = opus::remove_cvref_t<Traits>;
    using D_A = opus::fp8_t;
    using D_B = opus::fp8_t;
    using D_C = opus::bf16_t;
    using D_ACC = opus::fp32_t;
    using D_SF = unsigned char;
    using D_SF_PACK = unsigned int;
    namespace layout_9020 = opus_gemm_8wave_192x256_layout;
    namespace layout_9021 = opus_gemm_4wave_128x128_layout;

    const int wave_id = __builtin_amdgcn_readfirstlane(thread_id_x() / T::WARP_SIZE);
    const int lane_id = thread_id_x() % T::WARP_SIZE;
    const int wave_id_m = wave_id % T::T_M;
    const int wave_id_n = wave_id / T::T_M;
    const int row = block_id_y() * T::B_M;
    const int col = block_id_x() * T::B_N;
    const int batch_id = block_id_z();
    const int loops = kargs.k / T::B_K;

    auto g_a = make_gmem(reinterpret_cast<const D_A*>(kargs.ptr_a) + batch_id * kargs.stride_a_batch + row * kargs.stride_a,
                        static_cast<unsigned>((kargs.m - row) * kargs.stride_a));
    auto g_b = make_gmem(reinterpret_cast<const D_B*>(kargs.ptr_b) + batch_id * kargs.stride_b_batch + col * kargs.stride_b);
    auto g_c = make_gmem(reinterpret_cast<D_C*>(kargs.ptr_c) + batch_id * kargs.stride_c_batch + row * kargs.stride_c + col,
                        static_cast<unsigned>(((kargs.m - row) * kargs.stride_c - col) * sizeof(D_C)));
    auto g_sfa = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfa) + batch_id * kargs.stride_sfa_batch + row);
    auto g_sfb = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfb) + batch_id * kargs.stride_sfb_batch + (col / T::GROUP_N) * kargs.stride_sfb);

    const auto u_ga = make_layout_ga_scale<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_a);
    const auto u_sa = make_layout_sa_scale<T>(wave_id_m, wave_id_n);
    const auto u_ra = layout_9020::make_layout_ra_scale<T>(lane_id, wave_id_m);
    const auto u_gb = make_layout_gb_scale<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_b);
    const auto u_sb = make_layout_sb_scale<T>(wave_id_m, wave_id_n);
    const auto u_rb = make_layout_rb_scale<T>(lane_id, wave_id_n);
    const auto u_gsfa = layout_9021::make_layout_gsfa_scale<T, 0>(lane_id, wave_id_m, wave_id_n, kargs.stride_sfa);
    const auto u_ssfa = layout_9021::make_layout_ssfa_scale<T, 0>(lane_id, wave_id_m, wave_id_n);
    const auto u_rsfa = layout_9020::make_layout_rsfa_scale<T>(lane_id, wave_id_m);
    const auto u_gsfb = layout_9021::make_layout_gsfb_scale<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_sfb);
    const auto u_ssfb = layout_9021::make_layout_ssfb_scale<T>(lane_id, wave_id_m, wave_id_n);
    const auto u_rsfb = layout_9021::make_layout_rsfb_scale<T>();

    alignas(16) __shared__ char smem_matrix[T::LDS_BYTES];
    auto s_a = make_smem(reinterpret_cast<D_A*>(smem_matrix));
    auto s_b = make_smem(reinterpret_cast<D_B*>(smem_matrix + T::NUM_STAGES * T::A_STAGE));
    auto s_sfa = make_smem(reinterpret_cast<D_SF*>(smem_matrix + T::MATRIX_LDS_BYTES));
    auto s_sfb = make_smem(reinterpret_cast<D_SF*>(smem_matrix + T::MATRIX_LDS_BYTES + T::SFA_BYTES));

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

    const auto p_coord_c = opus::make_tuple(wave_id_m, lane_id % mma.grpn_c, wave_id_n, lane_id / mma.grpn_c);
    const auto u_gc = partition_layout_c<T::VEC_C>(mma, opus::make_tuple(kargs.stride_c, 1_I), p_coord_c);
    const auto gc_offsets = layout_to_offsets<T::VEC_C>(u_gc);
    const auto sfa_gmem_offsets = layout_to_offsets<T::VEC_SCALE_A>(u_gsfa);
    const auto sfa_smem_offsets = layout_to_offsets<T::VEC_SCALE_A>(u_ssfa);
    const auto rsfa_offsets = layout_to_offsets<1>(u_rsfa);
    const auto sfb_gmem_offsets = layout_to_offsets<1>(u_gsfb);
    const auto sfb_smem_offsets = layout_to_offsets<1>(u_ssfb);
    const auto rsfb_offsets = layout_to_offsets<1>(u_rsfb);
    D_SF_PACK v_sfa;
    D_SF_PACK v_sfb;

    auto ga_offset = [&](int tile_k) { return tile_k * number<T::B_K>{}; };
    auto gb_offset = [&](int tile_k) { return tile_k * number<T::B_K * T::W_N>{}; };
    auto sa_offset = [&](int stage) { return stage * T::A_STAGE; };
    auto sb_offset = [&](int stage) { return stage * T::B_STAGE; };
    auto gsfa_offset = [&](int panel_k_begin) { return panel_k_begin * kargs.stride_sfa; };
    auto gsfb_offset = [&](int panel_k_begin) { return panel_k_begin; };
    auto ssfa_offset = [&](int tile_k) { return (tile_k & (T::SCALE_PANEL - 1)) * T::B_M; };
    auto ssfb_offset = [&](int tile_k) { return tile_k & (T::SCALE_PANEL - 1); };

    auto issue_matrix_prefetch = [&](int stage, int tile_k) {
        async_load<T::VEC_A>(g_a, s_a.ptr, u_ga, u_sa + sa_offset(stage), ga_offset(tile_k));
        async_load<T::VEC_B>(g_b, s_b.ptr, u_gb, u_sb + sb_offset(stage), gb_offset(tile_k));
    };
    auto load_scale_panel = [&](int panel_k_begin) {
        const int sfa_offset = sfa_smem_offsets[0];
        const int sfa_group = sfa_offset / T::B_M;
        const int sfa_row = sfa_offset % T::B_M;
        if (sfa_offset < T::SFA_BYTES && panel_k_begin + sfa_group < loops) {
            const auto raw = layout_9021::load_sfa_vector<T>(
                g_sfa, sfa_gmem_offsets[0] + gsfa_offset(panel_k_begin),
                kargs.m - row - sfa_row);
            store<T::VEC_SCALE_A>(s_sfa, raw, sfa_offset);
        }
        const int sfb_group = sfb_smem_offsets[0];
        if (sfb_group < T::SCALE_PANEL && panel_k_begin + sfb_group < loops) {
            store<1>(s_sfb, load<1>(g_sfb, sfb_gmem_offsets[0] + gsfb_offset(panel_k_begin)), sfb_group);
        }
    };
    auto read_scales = [&](int tile_k) {
        v_sfa = 0;
        static_for<T::E_M>([&](auto m_i) {
            constexpr int m_repeat = decltype(m_i)::value;
            const D_SF_PACK value = load<1>(s_sfa, rsfa_offsets[m_repeat] + ssfa_offset(tile_k))[0];
            v_sfa |= value << (m_repeat * 8);
        });
        v_sfb = load<1>(s_sfb, rsfb_offsets[0] + ssfb_offset(tile_k))[0];
    };
    auto load_a_fragment = [&](auto m_i, int stage) {
        constexpr int m_repeat = decltype(m_i)::value;
        const auto offsets = layout_to_offsets<T::VEC_A>(u_ra + sa_offset(stage));
        static_for<T::A_CHUNKS_PER_FRAGMENT>([&](auto chunk_i) {
            constexpr int index = m_repeat * T::A_CHUNKS_PER_FRAGMENT + decltype(chunk_i)::value;
            set_slice(v_a, load<T::VEC_A>(s_a, offsets[index]),
                      number<index * T::VEC_A>{}, number<(index + 1) * T::VEC_A>{});
        });
    };
    auto load_b_fragment = [&](auto n_i, int stage) {
        constexpr int n_repeat = decltype(n_i)::value;
        const auto offsets = layout_to_offsets<T::VEC_B>(u_rb + sb_offset(stage));
        static_for<T::B_CHUNKS_PER_FRAGMENT>([&](auto chunk_i) {
            constexpr int index = n_repeat * T::B_CHUNKS_PER_FRAGMENT + decltype(chunk_i)::value;
            set_slice(v_b, load<T::VEC_B>(s_b, offsets[index]),
                      number<index * T::VEC_B>{}, number<(index + 1) * T::VEC_B>{});
        });
    };
    auto mma_scale_fragment = [&](auto m_i, auto n_i) {
        constexpr int m_repeat = decltype(m_i)::value;
        constexpr int n_repeat = decltype(n_i)::value;
        constexpr int c_index = m_repeat * T::E_N + n_repeat;
        const auto a = slice(v_a, number<m_repeat * ELEM_A>{}, number<(m_repeat + 1) * ELEM_A>{});
        const auto b = slice(v_b, number<n_repeat * ELEM_B>{}, number<(n_repeat + 1) * ELEM_B>{});
        auto c = slice(v_c, number<c_index * ELEM_C>{}, number<(c_index + 1) * ELEM_C>{});
        c = typename decltype(mma)::MMA{}(a, b, c,
                                        static_cast<int>(v_sfa), static_cast<int>(v_sfb),
                                        number<m_repeat>{}, number<0>{});
        set_slice(v_c, c, number<c_index * ELEM_C>{}, number<(c_index + 1) * ELEM_C>{});
    };

    // Prologue: one K128 tile and its scale panel.
    issue_matrix_prefetch(0, 0);
    load_scale_panel(0);
    s_waitcnt_vmcnt(0_I);
    s_waitcnt_lgkmcnt(0_I);
    __builtin_amdgcn_s_barrier();
    read_scales(0);
    static_for<T::E_M>([&](auto m_i) { load_a_fragment(m_i, 0); });
    static_for<T::E_N>([&](auto n_i) { load_b_fragment(n_i, 0); });
    s_waitcnt_lgkmcnt(0_I);

    int stage = 0;
    for (int tile_k = 0; tile_k + 1 < loops; ++tile_k) {
        const int next_stage = stage ^ 1;
        issue_matrix_prefetch(next_stage, tile_k + 1);
        static_for<T::E_M>([&](auto m_i) {
            static_for<T::E_N>([&](auto n_i) { mma_scale_fragment(m_i, n_i); });
        });
        s_waitcnt_vmcnt(0_I);
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_barrier();

        if (((tile_k + 1) & (T::SCALE_PANEL - 1)) == 0) {
            load_scale_panel(tile_k + 1);
            s_waitcnt_vmcnt(0_I);
            s_waitcnt_lgkmcnt(0_I);
            __builtin_amdgcn_s_barrier();
        }
        read_scales(tile_k + 1);
        static_for<T::E_M>([&](auto m_i) { load_a_fragment(m_i, next_stage); });
        static_for<T::E_N>([&](auto n_i) { load_b_fragment(n_i, next_stage); });
        s_waitcnt_lgkmcnt(0_I);
        stage = next_stage;
    }

    // Final K128 tile and direct BF16 output with the bounded C resource.
    static_for<T::E_M>([&](auto m_i) {
        static_for<T::E_N>([&](auto n_i) { mma_scale_fragment(m_i, n_i); });
    });
    static_for<T::E_M * T::E_N>([&](auto c_i) {
        constexpr int index = decltype(c_i)::value;
        const auto c = slice(v_c, number<index * ELEM_C>{}, number<(index + 1) * ELEM_C>{});
        store<T::VEC_C>(g_c, cast<D_C>(c), gc_offsets[index]);
    });
}
#endif
