#pragma once

#include "opus_gemm_traits_a8w8_mxscale_bpreshuffle_4wave_256x256_padded_m_gfx950.cuh"
#include "opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_4wave_gfx950.cuh"

#if !defined(__HIP_DEVICE_COMPILE__)
template<class Traits>
__global__ void gemm_a8w8_mxfp8_scale_4wave_256x256_padded_m_kernel(opus_gemm_mxscale_bpreshuffle_kargs_gfx950) {}
#elif defined(__gfx950__)
#if __has_cpp_attribute(clang::amdgpu_pin_agpr)

template<class Traits>
__global__ __launch_bounds__(256, 1) void gemm_a8w8_mxfp8_scale_4wave_256x256_padded_m_kernel(opus_gemm_mxscale_bpreshuffle_kargs_gfx950 kargs) {
    using namespace opus;

    using T = opus::remove_cvref_t<Traits>;
    using D_A = opus::fp8_t;
    using D_B = opus::fp8_t;
    using D_C = opus::bf16_t;
    using D_ACC = opus::fp32_t;
    using D_SF = unsigned char;
    using D_SF_PACK = unsigned int;

    // Tile and thread coordinates.
    int block_m = block_id_y();
    int block_n = block_id_x();
    if (((kargs.m | kargs.n) & 511) == 0) {
        const int old_m = block_m;
        block_m = (block_m & ~1) | (block_n & 1);
        block_n = (block_n & ~1) | (old_m & 1);
    }
    const int row = block_m * T::B_M;
    const int col = block_n * T::B_N;

    const int batch_id = block_id_z();
    const int wave_id = __builtin_amdgcn_readfirstlane(thread_id_x() / T::WARP_SIZE);
    const int lane_id = thread_id_x() % T::WARP_SIZE;

    // PAD_M uses a bounded view of the remaining real rows. Buffer loads
    // beyond the final row zero-fill LDS; no padded global tensor is needed.
    const unsigned int a_bytes = static_cast<unsigned int>((kargs.m - row) * kargs.stride_a);
    const unsigned int c_bytes = static_cast<unsigned int>(((kargs.m - row) * kargs.stride_c - col) * sizeof(D_C));
    // Matrix global-memory views.
    auto g_a = make_gmem(reinterpret_cast<const D_A*>(kargs.ptr_a) + batch_id * kargs.stride_a_batch + row * kargs.stride_a, a_bytes);
    auto g_b = make_gmem(reinterpret_cast<const D_B*>(kargs.ptr_b) + batch_id * kargs.stride_b_batch + col * kargs.stride_b);
    auto g_c = make_gmem(reinterpret_cast<D_C*>(kargs.ptr_c) + batch_id * kargs.stride_c_batch + row * kargs.stride_c + col, c_bytes);
    auto g_sfa = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfa) + batch_id * kargs.stride_sfa_batch + row, static_cast<unsigned int>(kargs.stride_sfa_batch - row));
    auto g_sfb = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfb) + batch_id * kargs.stride_sfb_batch + (col / T::GROUP_N + wave_id % T::SCALE_N_HALVES) * kargs.stride_sfb, static_cast<unsigned int>(kargs.stride_sfb));

    const int wave_id_m = wave_id % T::T_M;
    const int wave_id_n = wave_id / T::T_M;

    // Matrix layouts: global -> LDS -> registers.
    const auto u_ga = make_layout_ga_scale<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_a);
    const auto u_sa = make_layout_sa_scale<T>(wave_id_m, wave_id_n);
    const auto u_ra = make_layout_ra_scale<T>(lane_id, wave_id_m);
    const auto u_gb = make_layout_gb_scale<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_b);
    const auto u_sb = make_layout_sb_scale<T>(wave_id_m, wave_id_n);
    const auto u_rb = make_layout_rb_scale<T>(lane_id, wave_id_n);
    const auto u_gsfa = make_layout_gsfa_scale<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_sfa);
    const auto u_rsfa = make_layout_rsfa_scale<T>(lane_id, wave_id_m);
    const auto u_rsfa_pair = make_layout_rsfa_scale<T, T::VEC_SCALE_SF_PAIR>(lane_id, wave_id_m);
    const auto u_gsfb = make_layout_gsfb_scale<T>(lane_id, wave_id);
    const auto u_ssfb = make_layout_ssfb_scale<T>(lane_id, wave_id);
    const auto u_rsfb = make_layout_rsfb_scale<T>();
    const auto u_rsfb_pair = make_layout_rsfb_scale<T, T::VEC_SCALE_SF_PAIR>();

    // Matrix LDS; the same allocation is reused for the C epilogue.
    constexpr int smem_a_elem = T::smem_m_rep * (T::smem_linear_wave + T::smem_padding);
    constexpr int smem_b_elem = T::smem_n_rep * (T::smem_linear_wave + T::smem_padding);
    constexpr int matrix_lds_bytes = smem_a_elem * 4 * sizeof(D_A) + smem_b_elem * 4 * sizeof(D_B);
    alignas(16) __shared__ char smem_matrix[matrix_lds_bytes];
    auto s_a = make_smem(reinterpret_cast<D_A*>(smem_matrix));
    auto s_b = make_smem(reinterpret_cast<D_B*>(smem_matrix + smem_a_elem * 4 * sizeof(D_A)));
    auto s_c = make_smem(reinterpret_cast<D_C*>(smem_matrix));
    alignas(8) __shared__ char smem_sfa[T::SFA_PANEL_BYTES];
    alignas(8) __shared__ char smem_sfb[T::SFB_PANEL_BYTES];
    auto s_sfa = make_smem(reinterpret_cast<D_SF*>(smem_sfa));
    auto s_sfb = make_smem(reinterpret_cast<D_SF*>(smem_sfb));

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
    constexpr int A_CHUNKS_PER_FRAGMENT = sizeof(AFragment) / sizeof(AChunk);
    constexpr int B_CHUNKS_PER_FRAGMENT = sizeof(BFragment) / sizeof(BChunk);
    constexpr int A_DWORDS_PER_CHUNK = sizeof(AChunk) / sizeof(opus::u32_t);
    opus::array<AFragment, T::E_M> v_a[2];
    opus::array<BFragment, T::E_N> v_b;
    opus::array<BFragment, T::E_N> v_b_second;
    auto* a0_chunks = reinterpret_cast<AChunk*>(&v_a[0]);
    auto* a1_chunks = reinterpret_cast<AChunk*>(&v_a[1]);
    auto* b0_chunks = reinterpret_cast<BChunk*>(&v_b);
    auto* b1_chunks = reinterpret_cast<BChunk*>(&v_b_second);
    using CTile = opus::array<AccFragment, T::E_M * T::E_N>;
    using PinnedC01Tile = pinned_accumulator_array<AccFragment, T::E_M * T::E_N, C01_AGPR_BASE>;
    CTile c00{};
    PinnedC01Tile c01{};
    CTile c10{};
    CTile c11{};

    // C row stride in D_C elements while staging BF16 output in LDS.
    constexpr int c_lds_row_stride_elems = T::B_N + 8;

    D_SF_PACK v_sfa[2];
    D_SF_PACK v_sfb[T::SCALE_N_HALVES];
    opus::vector_t<D_SF_PACK, 2> v_sfb_next;
    opus::vector_t<D_SF_PACK, 2> v_sfa_next;

    constexpr int scale_panel_mask = T::SCALE_PANEL_K_CAPACITY - 1;
    auto ga_offset = [&](int half_tile_m, int tile_k) { return half_tile_m * T::HALF_B_M * kargs.stride_a + tile_k * T::B_K; };
    auto gb_offset = [&](int half_tile_n, int tile_k) { return half_tile_n * T::HALF_B_N * kargs.stride_b + tile_k * T::B_K * 16; };
    auto sa_offset = [&](int stage, int half_tile_m) { return (stage * 2 + half_tile_m) * smem_a_elem; };
    auto sb_offset = [&](int stage, int half_tile_n) { return (stage * 2 + half_tile_n) * smem_b_elem; };
    auto gsfa_offset = [&](int panel_k_begin) { return panel_k_begin * kargs.stride_sfa; };
    auto gsfb_offset = [&](int panel_k_begin) { return panel_k_begin; };
    auto ssfa_offset = [&](int k_tile, int half_tile_m) { return (k_tile & scale_panel_mask) * T::SFA_PANEL_PITCH + half_tile_m * T::VEC_SCALE_SF; };
    auto ssfb_offset = [&](int k_tile, int half_tile_n) { return (k_tile & scale_panel_mask) * T::SCALE_N_HALVES * T::VEC_SCALE_SF + half_tile_n * T::VEC_SCALE_SF; };

    // A/B K-tile buffer_load from global memory to LDS. Each producer wave
    const bool prefetches_a = wave_id_n == 0;
    auto g_matrix_prefetch = prefetches_a ? g_a : g_b; // g_a: wave_id 0/1  g_b: wave_id 2/3
    constexpr int a_k_lanes = T::B_K / T::VEC_A; // 128 / 16 = 8
    constexpr int matrix_lds_pitch = T::smem_linear_wave + T::smem_padding; // 1024 + 32
    const int matrix_lane_offset = prefetches_a ? (wave_id_m * T::HALF_B_M + (lane_id / a_k_lanes) * 2) * kargs.stride_a + (lane_id % a_k_lanes) * T::VEC_A : wave_id_m * T::HALF_B_N * kargs.stride_b + lane_id * T::VEC_B;
    const int matrix_pair_stride = 16 * kargs.stride_a;
    const int matrix_odd_stride = prefetches_a ? kargs.stride_a : T::WARP_SIZE * T::VEC_B;
    const int matrix_tile_stride = prefetches_a ? T::B_K : T::B_K * T::VEC_B; // A: 128 / B: 128 * 16
    auto* matrix_lds_base = (prefetches_a ? s_a.ptr : s_b.ptr) + wave_id_m * smem_a_elem; // A or B and half

    // 128 x 128, every wave need 16 buffer_load_dwordx4, divided 4 group, every group divided 4 local, each local's 8 rows are staggered between odd and even, mimicking 4-wave load
    opus::vector_t<int, 4> matrix_voffsets;
    opus::static_for<4>([&](auto local_i) { // 0/1/2/3
        constexpr int local = decltype(local_i)::value;
        constexpr int immediate = local * matrix_lds_pitch; // 0/1/2/3 * (1024 + 32)
        int voffset = matrix_lane_offset + (local / 2) * matrix_pair_stride + (local % 2) * matrix_odd_stride - immediate;
        asm volatile("" : "+v"(voffset));
        matrix_voffsets[local] = voffset;
    });

    // Issue one of the 16 producer-wave requests that copy a K128 matrix half tile directly from global memory to an LDS ping-pong slot.
    // issue_i: buffer_load request index [0, 15], group=issue/4 and local=issue%4. Four requests form one group.
    // matrix_stage: 0 or 1
    // tile_offset: global K tile offset
    auto issue_matrix_prefetch = [&](auto issue_i, int matrix_stage, int tile_offset) {
        constexpr int issue = decltype(issue_i)::value;
        constexpr int local = issue % 4;
        constexpr int group = issue / 4;
        constexpr int immediate = local * matrix_lds_pitch;
        auto* dst = matrix_lds_base + matrix_stage * 2 * smem_a_elem + group * 4 * matrix_lds_pitch;
        int voffset = static_cast<int>(matrix_voffsets[local]) + __builtin_amdgcn_readfirstlane(tile_offset + group * 2 * matrix_pair_stride);
        asm volatile("" : "+v"(voffset));
        async_load<16>(
            g_matrix_prefetch,
            reinterpret_cast<void*>(reinterpret_cast<__UINTPTR_TYPE__>(dst)),
            voffset, 0, opus::number<immediate>{}, opus::number<0>{});
    };

    // Load Scale_A
    const int scale_k_groups = static_cast<unsigned int>(kargs.k) / T::GROUP_K;
    constexpr int sfa_lds_word_stride = T::VEC_SF * (T::B_M / T::HALF_B_M) * T::E_M; // 4 * （256 / 128）* 4 = 32
    const auto sfa_gmem_offsets = opus::layout_to_offsets<T::VEC_SCALE_A>(u_gsfa);
    opus::vector_t<D_SF, T::VEC_SCALE_A> sfa_panel_raw[T::SFA_PASSES_PER_CACHE_PANEL];

    // Scale_A Global to VGPR
    auto load_sfa_panel = [&](int panel_k_begin) { // K<= 8192 panel_k_begin = 0
        opus::static_for<T::SFA_PASSES_PER_CACHE_PANEL>([&](auto pass_i) { // 0/1/2/3
            constexpr int pass = decltype(pass_i)::value;
            const int first_k_group = panel_k_begin + pass * T::SFA_K_COLUMNS_PER_PASS + wave_id * T::SFA_K_COLUMNS_PER_WAVE;
            // 0 + 0/1/2/3 * 16 + 0/1/2/3 * 4
            // 8192/128 = 64 K groups, divided into 4 passes, with 16 K groups per pass. Each pass has 4 waves, and each wave consecutively loads 4 groups.
            // Inactive K passes are never published. Defining them here keeps
            // their old values from staying live across the matrix loop.
            sfa_panel_raw[pass] = {};
            if (first_k_group < scale_k_groups) {
                const int scale_row = ((lane_id & 8) | ((lane_id & 3) << 1) | ((lane_id & 4) >> 2)) * T::VEC_SCALE_A;
                if (row + scale_row < kargs.m) {
                    sfa_panel_raw[pass] = load<T::VEC_SCALE_A>(g_sfa, sfa_gmem_offsets[pass] + gsfa_offset(panel_k_begin));
                }
            }
        });
    };

    // Scale_A VGPR transpose and to lds
    auto publish_sfa_panel = [&](int panel_k_begin) { // K<= 8192 panel_k_begin = 0
        // Rebuild the SFA LDS write layout at each publish site instead of
        // carrying its derived address base through the matrix mainloop.
        int publish_lane_id = lane_id;
        asm volatile("" : "+v"(publish_lane_id));
        const auto publish_u_ssfa = make_layout_ssfa_scale<T>(publish_lane_id, wave_id_m, wave_id_n);
        const auto publish_sfa_smem_offsets = opus::layout_to_offsets<T::VEC_SF>(publish_u_ssfa);

        const int lane_in_quad = publish_lane_id % 4; // lane in quad
        // quad 0: lane 0~3
        // quad 1: lane 4~7
        // ....
        /// quad 15: lane 60~63
        const unsigned int pair_byte_select = (lane_in_quad % 2) ? 0x03070105u : 0x06020400u;
        const unsigned int quad_byte_select = (lane_in_quad / 2) ? 0x03020706u : 0x05040100u;

        auto transpose_sfa_word_4x4 = [&](D_SF_PACK lane_word) {
            const D_SF_PACK adjacent_lane = opus::mov_dpp(lane_word, opus::number<0xb1>{});
            const D_SF_PACK paired_bytes = __builtin_amdgcn_perm(adjacent_lane, lane_word, pair_byte_select);
            const D_SF_PACK opposite_pair = opus::mov_dpp(paired_bytes, opus::number<0x4e>{});
            return __builtin_amdgcn_perm(opposite_pair, paired_bytes, quad_byte_select);
        };
        // Example
        //                byte0   byte1   byte2   byte3
        // lane 0         M0      M1      M2      M3                lane 0：[M0, M32, M64, M96]
        // lane 1         M32     M33     M34     M35      ---->    lane 1：[M1, M33, M65, M97]
        // lane 2         M64     M65     M66     M67               lane 2：[M2, M34, M66, M98]
        // lane 3         M96     M97     M98     M99               lane 3：[M3, M35, M67, M99]

        opus::static_for<T::SFA_PASSES_PER_CACHE_PANEL>([&](auto pass_i) { // 0/1/2/3
            constexpr int pass = decltype(pass_i)::value;
            const int first_k_group = panel_k_begin + pass * T::SFA_K_COLUMNS_PER_PASS + wave_id * T::SFA_K_COLUMNS_PER_WAVE;
            // 0 + 0/1/2/3 * 16 + 0/1/2/3 * 4
            if (first_k_group < scale_k_groups) {
                const auto raw_words = __builtin_bit_cast(opus::vector_t<D_SF_PACK, 4>, sfa_panel_raw[pass]); // 16 = 4 x 4
                opus::static_for<4>([&](auto word_i) {
                    constexpr int word = decltype(word_i)::value;
                    // The 4x4 byte transpose happens here, entirely in VGPRs.
                    const D_SF_PACK packed_scales = transpose_sfa_word_4x4(raw_words[word]);
                    store<T::VEC_SF>(s_sfa, __builtin_bit_cast(opus::vector_t<D_SF, T::VEC_SF>, packed_scales), publish_sfa_smem_offsets[pass] + word * sfa_lds_word_stride);
                });
            }
        });
    };

    // Refill the 64-K128 SFA and SFB LDS panels together at each panel boundary, sharing one synchronized LDS handoff.
    auto refill_scale_panel = [&](int next_tile) {
        if ((next_tile & scale_panel_mask) == 0) {
            s_waitcnt_lgkmcnt(0_I);
            __builtin_amdgcn_s_barrier();
            load_sfa_panel(next_tile);
            D_SF_PACK raw_b = 0;
            if (wave_id < T::SCALE_N_HALVES) {
                const auto raw = load<1>(g_sfb, u_gsfb + gsfb_offset(next_tile));
                raw_b = static_cast<D_SF_PACK>(raw[0]);
            }
            s_waitcnt_vmcnt(0_I);
            publish_sfa_panel(next_tile);
            if (wave_id < T::SCALE_N_HALVES) {
                const D_SF_PACK packed = raw_b;
                store<T::VEC_SF>(s_sfb, __builtin_bit_cast(opus::vector_t<D_SF, T::VEC_SF>, packed), u_ssfb);
            }
            s_waitcnt_lgkmcnt(0_I);
            __builtin_amdgcn_s_barrier();
        }
    };

    int stage = 0;
    int tile = 0;
    const int loops = static_cast<unsigned int>(kargs.k) / T::B_K;

    // ===== Prologue =====
    async_load<T::VEC_A>(g_a, s_a.ptr, u_ga + ga_offset(0, 0), u_sa + sa_offset(0, 0));
    async_load<T::VEC_A>(g_a, s_a.ptr, u_ga + ga_offset(1, 0), u_sa + sa_offset(0, 1));
    load_sfa_panel(0);

    D_SF_PACK panel_sfb_raw = 0;
    if (wave_id < T::SCALE_N_HALVES) {
        const auto raw = load<1>(g_sfb, u_gsfb + gsfb_offset(0));
        panel_sfb_raw = static_cast<D_SF_PACK>(raw[0]);
    }
    async_load<T::VEC_B>(g_b, s_b.ptr, u_gb, u_sb + sb_offset(0, 0), gb_offset(0, 0));
    async_load<T::VEC_B>(g_b, s_b.ptr, u_gb, u_sb + sb_offset(0, 1), gb_offset(1, 0));

    publish_sfa_panel(0);
    if (wave_id < T::SCALE_N_HALVES) {
        const D_SF_PACK packed = panel_sfb_raw;
        store<T::VEC_SF>(s_sfb, __builtin_bit_cast(opus::vector_t<D_SF, T::VEC_SF>, packed), u_ssfb);
    }

    s_waitcnt_vmcnt(0_I);
    s_waitcnt_lgkmcnt(0_I);
    __builtin_amdgcn_s_barrier();

    // Interleave K1 requests with independent K0 LDS reads.
    if (loops > 1) { opus::static_for<8>([&](auto initial_issue) { issue_matrix_prefetch(initial_issue, 1, matrix_tile_stride); });}

    v_sfa[0] = __builtin_bit_cast(D_SF_PACK, load<T::VEC_SCALE_SF>(s_sfa, u_rsfa + ssfa_offset(0, 0)));
    v_sfa[1] = __builtin_bit_cast(D_SF_PACK, load<T::VEC_SCALE_SF>(s_sfa, u_rsfa + ssfa_offset(0, 1)));
    v_sfb[0] = __builtin_bit_cast(D_SF_PACK, load<T::VEC_SCALE_SF>(s_sfb, u_rsfb + ssfb_offset(0, 0)));
    v_sfb[1] = __builtin_bit_cast(D_SF_PACK, load<T::VEC_SCALE_SF>(s_sfb, u_rsfb + ssfb_offset(0, 1)));
    load_operand_chunks_pinned<T::VEC_B, T::E_N * B_CHUNKS_PER_FRAGMENT, B0_AGPR_BASE>(b0_chunks, s_b, u_rb + sb_offset(stage, 0));
    load_operand_chunks_pinned<T::VEC_A, T::E_M * A_CHUNKS_PER_FRAGMENT, A0_AGPR_BASE>(a0_chunks, s_a, u_ra + sa_offset(stage, 0));

    if (loops > 1) {
        opus::static_for<8>([&](auto initial_issue) {
            using Issue = opus::number<decltype(initial_issue)::value + 8>;
            issue_matrix_prefetch(Issue{}, 1, matrix_tile_stride);
        });
    }
    load_operand_chunks_pinned<T::VEC_A, T::E_M * A_CHUNKS_PER_FRAGMENT, A1_AGPR_BASE>(a1_chunks, s_a, u_ra + sa_offset(stage, 1));
    load_operand_chunks_pinned<T::VEC_B, T::E_N * B_CHUNKS_PER_FRAGMENT, B1_AGPR_BASE>(b1_chunks, s_b, u_rb + sb_offset(stage, 1));
    __builtin_amdgcn_s_setprio(1);

    // ===== Main loop =====
#pragma unroll 4
    for (; tile + 2 < loops; ++tile) {
        const int k_tile = tile;
        asm volatile("" : "+v"(matrix_voffsets[0]), "+v"(matrix_voffsets[1]), "+v"(matrix_voffsets[2]), "+v"(matrix_voffsets[3]));
        refill_scale_panel(k_tile + 1);
        const int next_stage = stage ^ 1;
        const int future_tile = k_tile + 2;
        int future_matrix_offset = __builtin_amdgcn_readfirstlane(future_tile * matrix_tile_stride);
        asm volatile("" : "+s"(future_matrix_offset));

        // A half 0 x B half 0 -> C[0][0]
        mma_scale_group<T, 0, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
        v_sfa_next = __builtin_bit_cast(opus::vector_t<D_SF_PACK, 2>, load<T::VEC_SCALE_SF_PAIR>(s_sfa, u_rsfa_pair + ssfa_offset(k_tile + 1, 0)));
        mma_scale_group<T, 2, 3>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
        __builtin_amdgcn_sched_barrier(0);
        s_waitcnt_vmcnt(0_I);
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_sched_barrier(0);
        mma_scale_group<T, 5, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
        __builtin_amdgcn_s_barrier();

        mma_scale_group<T, 6, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
        mma_scale_group<T, 7, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
        issue_matrix_prefetch(opus::number<0>{}, stage, future_matrix_offset);
        __builtin_amdgcn_sched_group_barrier(0x008, 3, 0);
        __builtin_amdgcn_sched_group_barrier(0x800, 1, 0);

        mma_scale_group<T, 8, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
        issue_matrix_prefetch(opus::number<1>{}, stage, future_matrix_offset);
        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0);
        __builtin_amdgcn_sched_group_barrier(0x800, 1, 0);

        mma_scale_group<T, 9, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
        mma_scale_group<T, 10, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
        issue_matrix_prefetch(opus::number<2>{}, stage, future_matrix_offset);
        __builtin_amdgcn_sched_group_barrier(0x008, 2, 0);
        __builtin_amdgcn_sched_group_barrier(0x800, 1, 0);

        mma_scale_group<T, 11, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
        mma_scale_group<T, 12, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
        issue_matrix_prefetch(opus::number<3>{}, stage, future_matrix_offset);
        __builtin_amdgcn_sched_group_barrier(0x008, 2, 0);
        __builtin_amdgcn_sched_group_barrier(0x800, 1, 0);

        mma_scale_group<T, 13, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
        mma_scale_group<T, 14, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
        issue_matrix_prefetch(opus::number<4>{}, stage, future_matrix_offset);
        mma_scale_group<T, 15, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
        __builtin_amdgcn_sched_group_barrier(0x008, 2, 0);
        __builtin_amdgcn_sched_group_barrier(0x800, 1, 0);
        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0);
        __builtin_amdgcn_sched_barrier(0);

        // A half 1 x B half 0 -> C[1][0]
        mma_scale_group<T, 0, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
        issue_matrix_prefetch(opus::number<5>{}, stage, future_matrix_offset);
        __builtin_amdgcn_sched_group_barrier(0x008, 1, 1);
        __builtin_amdgcn_sched_group_barrier(0x800, 1, 1);

        mma_scale_group<T, 1, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
        mma_scale_group<T, 2, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
        issue_matrix_prefetch(opus::number<6>{}, stage, future_matrix_offset);
        __builtin_amdgcn_sched_group_barrier(0x008, 2, 1);
        __builtin_amdgcn_sched_group_barrier(0x800, 1, 1);

        mma_scale_group<T, 3, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
        v_sfb_next = __builtin_bit_cast(opus::vector_t<D_SF_PACK, 2>, load<T::VEC_SCALE_SF_PAIR>(s_sfb, u_rsfb_pair + ssfb_offset(k_tile + 1, 0)));
        mma_scale_group<T, 4, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
        issue_matrix_prefetch(opus::number<7>{}, stage, future_matrix_offset);
        __builtin_amdgcn_sched_group_barrier(0x008, 1, 1);
        __builtin_amdgcn_sched_group_barrier(0x100, 1, 1);
        __builtin_amdgcn_sched_group_barrier(0x008, 1, 1);
        __builtin_amdgcn_sched_group_barrier(0x800, 1, 1);

        mma_scale_group<T, 5, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
        mma_scale_group<T, 6, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
        issue_matrix_prefetch(opus::number<8>{}, stage, future_matrix_offset);
        __builtin_amdgcn_sched_group_barrier(0x008, 2, 1);
        __builtin_amdgcn_sched_group_barrier(0x800, 1, 1);

        mma_scale_group<T, 7, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
        mma_scale_group<T, 8, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
        issue_matrix_prefetch(opus::number<9>{}, stage, future_matrix_offset);
        __builtin_amdgcn_sched_group_barrier(0x008, 2, 1);
        __builtin_amdgcn_sched_group_barrier(0x800, 1, 1);

        mma_scale_group<T, 9, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
        mma_scale_group<T, 10, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
        issue_matrix_prefetch(opus::number<10>{}, stage, future_matrix_offset);
        __builtin_amdgcn_sched_group_barrier(0x008, 2, 1);
        __builtin_amdgcn_sched_group_barrier(0x800, 1, 1);

        mma_scale_group<T, 11, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
        mma_scale_group<T, 12, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
        issue_matrix_prefetch(opus::number<11>{}, stage, future_matrix_offset);
        __builtin_amdgcn_sched_group_barrier(0x008, 2, 1);
        __builtin_amdgcn_sched_group_barrier(0x800, 1, 1);

        mma_scale_group<T, 13, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
        mma_scale_group<T, 14, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
        mma_scale_group<T, 15, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
        load_operand_fragment_pinned<T::VEC_B, 0>(opus::number<64>{}, v_b[0], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 0)));
        __builtin_amdgcn_sched_group_barrier(0x008, 3, 1);
        __builtin_amdgcn_sched_group_barrier(0x100, 2, 1);
        __builtin_amdgcn_sched_barrier(0);


        // A half 0 x B half 1 -> C[0][1]
        mma_scale_group<T, 0, 1>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
        issue_matrix_prefetch(opus::number<12>{}, stage, future_matrix_offset);
        __builtin_amdgcn_sched_group_barrier(0x008, 1, 2);
        __builtin_amdgcn_sched_group_barrier(0x800, 1, 2);

        mma_scale_group<T, 1, 1>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
        load_operand_fragment_pinned<T::VEC_B, 2>(opus::number<72>{}, v_b[1], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 0)));
        __builtin_amdgcn_sched_group_barrier(0x008, 1, 2);
        __builtin_amdgcn_sched_group_barrier(0x100, 2, 2);

        mma_scale_group<T, 2, 1>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
        mma_scale_group<T, 3, 1>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
        load_operand_fragment_pinned<T::VEC_A, 0>(opus::number<0>{}, v_a[0][0], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0)));
        load_operand_fragment_pinned<T::VEC_B, 4>(opus::number<80>{}, v_b[2], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 0)));
        __builtin_amdgcn_sched_group_barrier(0x008, 2, 2);
        __builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
        __builtin_amdgcn_sched_group_barrier(0x100, 2, 2);

        mma_scale_group<T, 4, 1>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
        issue_matrix_prefetch(opus::number<13>{}, stage, future_matrix_offset);
        __builtin_amdgcn_sched_group_barrier(0x008, 1, 2);
        __builtin_amdgcn_sched_group_barrier(0x800, 1, 2);

        mma_scale_group<T, 5, 1>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
        load_operand_fragment_pinned<T::VEC_B, 6>(opus::number<88>{}, v_b[3], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 0)));
        __builtin_amdgcn_sched_group_barrier(0x008, 1, 2);
        __builtin_amdgcn_sched_group_barrier(0x100, 2, 2);

        v_sfb[0] = v_sfb_next[0];

        issue_matrix_prefetch(opus::number<14>{}, stage, future_matrix_offset);
        mma_scale_group<T, 6, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
        load_operand_fragment_pinned<T::VEC_A, 2>(opus::number<8>{}, v_a[0][1], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0)));
        __builtin_amdgcn_sched_group_barrier(0x800, 1, 2);
        __builtin_amdgcn_sched_group_barrier(0x008, 2, 2);
        __builtin_amdgcn_sched_group_barrier(0x100, 2, 2);

        issue_matrix_prefetch(opus::number<15>{}, stage, future_matrix_offset);
        mma_scale_group<T, 8, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
        mma_scale_group<T, 10, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
        load_operand_fragment_pinned<T::VEC_A, 4>(opus::number<16>{}, v_a[0][2], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0)));
        __builtin_amdgcn_sched_group_barrier(0x800, 1, 2);
        __builtin_amdgcn_sched_group_barrier(0x008, 4, 2);
        __builtin_amdgcn_sched_group_barrier(0x100, 2, 2);

        mma_scale_group<T, 12, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
        mma_scale_group<T, 14, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);

        v_sfa[0] = v_sfa_next[0];
        load_operand_fragment_pinned<T::VEC_A, 6>(opus::number<24>{}, v_a[0][3], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0)));
        __builtin_amdgcn_sched_group_barrier(0x008, 4, 2);
        __builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
        __builtin_amdgcn_sched_barrier(0);

        // A half 1 x B half 1 -> C[1][1]
        mma_scale_group<T, 0, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
        mma_scale_group<T, 2, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
        load_operand_fragment_pinned<T::VEC_A, 0>(opus::number<32>{}, v_a[1][0], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 1)));
        __builtin_amdgcn_sched_group_barrier(0x008, 4, 3);
        __builtin_amdgcn_sched_group_barrier(0x100, 2, 3);

        mma_scale_group<T, 4, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);

        const BFragment b1_n3_next = load_operand_fragment_staged<T::VEC_B, 6, BFragment>(
            s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 1)));

        mma_scale_group<T, 5, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
        mma_scale_group<T, 6, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
        load_operand_fragment_pinned<T::VEC_A, 2>(opus::number<40>{}, v_a[1][1], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 1)));
        const AFragment a1_m3_next = load_operand_fragment_staged<T::VEC_A, 6, AFragment>(
            s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 1)));
        __builtin_amdgcn_sched_group_barrier(0x008, 1, 3);
        __builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
        __builtin_amdgcn_sched_group_barrier(0x008, 3, 3);
        __builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
        __builtin_amdgcn_sched_group_barrier(0x100, 2, 3);

        mma_scale_group<T, 8, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
        mma_scale_group<T, 12, 1, C11_TAIL_AGPR_BASE, 11>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
        load_operand_fragment_pinned<T::VEC_B, 0>(opus::number<96>{}, v_b_second[0], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 1)));
        __builtin_amdgcn_sched_group_barrier(0x008, 2, 3);
        __builtin_amdgcn_sched_group_barrier(0x100, 2, 3);

        mma_scale_group<T, 9, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
        mma_scale_group<T, 13, 1, C11_TAIL_AGPR_BASE, 11>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
        load_operand_fragment_pinned<T::VEC_B, 2>(opus::number<104>{}, v_b_second[1], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 1)));
        __builtin_amdgcn_sched_group_barrier(0x008, 2, 3);
        __builtin_amdgcn_sched_group_barrier(0x100, 2, 3);

        mma_scale_group<T, 10, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
        mma_scale_group<T, 14, 1, C11_TAIL_AGPR_BASE, 11>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
        load_operand_fragment_pinned<T::VEC_B, 4>(opus::number<112>{}, v_b_second[2], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 1)));
        __builtin_amdgcn_sched_group_barrier(0x008, 2, 3);
        __builtin_amdgcn_sched_group_barrier(0x100, 2, 3);

        mma_scale_group<T, 11, 1, C11_TAIL_AGPR_BASE, 11>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
        load_operand_fragment_pinned<T::VEC_A, 4>(opus::number<48>{}, v_a[1][2], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 1)));
        mma_scale_group<T, 15, 1, C11_TAIL_AGPR_BASE, 11>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
        __builtin_amdgcn_sched_group_barrier(0x008, 1, 3);
        __builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
        __builtin_amdgcn_sched_group_barrier(0x008, 1, 3);

        [[clang::amdgpu_pin_agpr(A1_AGPR_BASE + 3 * sizeof(AFragment) / sizeof(opus::u32_t))]] v_a[1][3] = a1_m3_next;
        [[clang::amdgpu_pin_agpr(B1_AGPR_BASE + 3 * sizeof(BFragment) / sizeof(opus::u32_t))]] v_b_second[3] = b1_n3_next;
        __builtin_amdgcn_sched_barrier(0x004);

        v_sfb[1] = v_sfb_next[1];
        v_sfa[1] = v_sfa_next[1];
        stage = next_stage;
    }
    // ===== Epilogue =====
    if (loops > 1) {
        refill_scale_panel(tile + 1);
        const int next_stage = stage ^ 1;
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_setprio(1);

        // A half 0 x B half 0 -> C[0][0]
        mma_scale_group<T, 0, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
        v_sfa_next = __builtin_bit_cast(opus::vector_t<D_SF_PACK, 2>, load<T::VEC_SCALE_SF_PAIR>(s_sfa, u_rsfa_pair + ssfa_offset(tile + 1, 0)));
        __builtin_amdgcn_sched_group_barrier(0x008, 2, 0);
        __builtin_amdgcn_sched_group_barrier(0x100, 1, 0);

        mma_scale_group<T, 2, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
        mma_scale_group<T, 4, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
        __builtin_amdgcn_sched_group_barrier(0x008, 3,0);
        __builtin_amdgcn_s_setprio(0);
        s_waitcnt_vmcnt(0_I);
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_s_setprio(1);

        mma_scale_group<T, 5, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0);

        mma_scale_group<T, 6, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
        mma_scale_group<T, 8, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
        mma_scale_group<T, 10, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
        mma_scale_group<T, 12, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
        mma_scale_group<T, 14, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
        __builtin_amdgcn_sched_group_barrier(0x008, 10, 0);

        // A half 1 x B half 0 -> C[1][0]
        mma_scale_group<T, 0, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
        mma_scale_group<T, 2, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
        v_sfb_next = __builtin_bit_cast(opus::vector_t<D_SF_PACK, 2>, load<T::VEC_SCALE_SF_PAIR>(s_sfb, u_rsfb_pair + ssfb_offset(tile + 1, 0)));
        __builtin_amdgcn_sched_group_barrier(0x008, 4, 1);
        __builtin_amdgcn_sched_group_barrier(0x100, 1, 1);

        mma_scale_group<T, 4, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
        mma_scale_group<T, 6, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
        mma_scale_group<T, 8, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
        mma_scale_group<T, 10, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
        mma_scale_group<T, 12, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);

        AChunk a0_m0_next_0;
        AChunk a0_m0_next_1;
        [[clang::amdgpu_pin_agpr(A0_M0_STAGE_AGPR_BASE)]] a0_m0_next_0 = load<T::VEC_A>(s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0))[0]);
        [[clang::amdgpu_pin_agpr(A0_M0_STAGE_AGPR_BASE + A_DWORDS_PER_CHUNK)]] a0_m0_next_1 = load<T::VEC_A>(s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0))[1]);
        __builtin_amdgcn_sched_group_barrier(0x008, 10, 1);
        __builtin_amdgcn_sched_group_barrier(0x100, 2, 1);

        mma_scale_group<T, 14, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
        load_operand_fragment_pinned<T::VEC_B, 0>(opus::number<64>{}, v_b[0], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 0)));
        __builtin_amdgcn_sched_group_barrier(0x008, 2, 1);
        __builtin_amdgcn_sched_group_barrier(0x100, 2, 1);
        __builtin_amdgcn_sched_barrier(0);

        // A half 0 x B half 1 -> C[0][1]
        mma_scale_group<T, 0, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
        load_operand_fragment_pinned<T::VEC_B, 2>(opus::number<72>{}, v_b[1], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 0)));
        __builtin_amdgcn_sched_group_barrier(0x008, 2, 2);
        __builtin_amdgcn_sched_group_barrier(0x100, 2, 2);

        mma_scale_group<T, 2, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
        load_operand_fragment_pinned<T::VEC_B, 4>(opus::number<80>{}, v_b[2], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 0)));
        __builtin_amdgcn_sched_group_barrier(0x008, 2, 2);
        __builtin_amdgcn_sched_group_barrier(0x100, 2, 2);

        AFragment a0_m0_joined;
        opus::set_slice(a0_m0_joined, a0_m0_next_0, opus::number<0>{}, opus::number<16>{});
        opus::set_slice(a0_m0_joined, a0_m0_next_1, opus::number<16>{}, opus::number<32>{});
        [[clang::amdgpu_pin_agpr(A0_AGPR_BASE)]] v_a[0][0] = a0_m0_joined;
        __builtin_amdgcn_s_setprio(1);

        mma_scale_group<T, 4, 1>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
        mma_scale_group<T, 5, 1>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
        load_operand_fragment_pinned<T::VEC_B, 6>(opus::number<88>{}, v_b[3], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 0)));
        __builtin_amdgcn_sched_group_barrier(0x008, 2, 2);
        __builtin_amdgcn_sched_group_barrier(0x100, 2, 2);

        v_sfb[0] = v_sfb_next[0];

        mma_scale_group<T, 6, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
        load_operand_fragment_pinned<T::VEC_A, 2>(opus::number<8>{}, v_a[0][1], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0)));
        __builtin_amdgcn_sched_group_barrier(0x008, 2, 2);
        __builtin_amdgcn_sched_group_barrier(0x100, 2, 2);

        mma_scale_group<T, 8, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
        mma_scale_group<T, 10, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
        load_operand_fragment_pinned<T::VEC_A, 4>(opus::number<16>{}, v_a[0][2], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0)));
        __builtin_amdgcn_sched_group_barrier(0x008, 4, 2);
        __builtin_amdgcn_sched_group_barrier(0x100, 2, 2);

        mma_scale_group<T, 12, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
        mma_scale_group<T, 14, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);

        v_sfa[0] = v_sfa_next[0];
        load_operand_fragment_pinned<T::VEC_A, 6>(opus::number<24>{}, v_a[0][3], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0)));
        __builtin_amdgcn_sched_group_barrier(0x008, 4, 2);
        __builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
        __builtin_amdgcn_sched_barrier(0);

        // A half 1 x B half 1 -> C[1][1]
        mma_scale_group<T, 0, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
        mma_scale_group<T, 2, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
        load_operand_fragment_pinned<T::VEC_A, 0>(opus::number<32>{}, v_a[1][0], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 1)));
        __builtin_amdgcn_sched_group_barrier(0x008, 4, 3);
        __builtin_amdgcn_sched_group_barrier(0x100, 2, 3);

        mma_scale_group<T, 4, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
        mma_scale_group<T, 6, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
        load_operand_fragment_pinned<T::VEC_A, 2>(opus::number<40>{}, v_a[1][1], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 1)));
        __builtin_amdgcn_sched_group_barrier(0x008, 4, 3);
        __builtin_amdgcn_sched_group_barrier(0x100, 2, 3);

        mma_scale_group<T, 8, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
        mma_scale_group<T, 12, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
        load_operand_fragment_pinned<T::VEC_B, 0>(opus::number<96>{}, v_b_second[0], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 1)));
        __builtin_amdgcn_sched_group_barrier(0x008, 2, 3);
        __builtin_amdgcn_sched_group_barrier(0x100, 2, 3);

        mma_scale_group<T, 9, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
        mma_scale_group<T, 13, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
        load_operand_fragment_pinned<T::VEC_B, 2>(opus::number<104>{}, v_b_second[1], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 1)));
        __builtin_amdgcn_sched_group_barrier(0x008, 2, 3);
        __builtin_amdgcn_sched_group_barrier(0x100, 2, 3);

        mma_scale_group<T, 10, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
        mma_scale_group<T, 14, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
        load_operand_fragment_pinned<T::VEC_B, 4>(opus::number<112>{}, v_b_second[2], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 1)));
        __builtin_amdgcn_sched_group_barrier(0x008, 2, 3);
        __builtin_amdgcn_sched_group_barrier(0x100, 2, 3);

        mma_scale_group<T, 11, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
        load_operand_fragment_pinned<T::VEC_A, 4>(opus::number<48>{}, v_a[1][2], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 1)));
        mma_scale_group<T, 15, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
        load_operand_fragment_pinned<T::VEC_A, 6>(opus::number<56>{}, v_a[1][3], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 1)));
        load_operand_fragment_pinned<T::VEC_B, 6>(opus::number<120>{}, v_b_second[3], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 1)));
        __builtin_amdgcn_sched_group_barrier(0x008, 1, 3);
        __builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
        __builtin_amdgcn_sched_group_barrier(0x008, 1, 3);
        __builtin_amdgcn_sched_group_barrier(0x100, 4, 3);
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_s_setprio(0);
        v_sfb[1] = v_sfb_next[1];
        v_sfa[1] = v_sfa_next[1];
        stage = next_stage;
    }

    // Final K tile
    __builtin_amdgcn_s_setprio(1);
    mma_scale_group<T, 0, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
    mma_scale_group<T, 2, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
    mma_scale_group<T, 4, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
    mma_scale_group<T, 6, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
    mma_scale_group<T, 8, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
    mma_scale_group<T, 10, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
    mma_scale_group<T, 12, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
    mma_scale_group<T, 14, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
    mma_scale_group<T, 0, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
    mma_scale_group<T, 2, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
    mma_scale_group<T, 4, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
    mma_scale_group<T, 6, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
    mma_scale_group<T, 8, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
    mma_scale_group<T, 10, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
    mma_scale_group<T, 12, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
    mma_scale_group<T, 14, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
    mma_scale_group<T, 0, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
    mma_scale_group<T, 2, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
    mma_scale_group<T, 4, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
    mma_scale_group<T, 6, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
    mma_scale_group<T, 8, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
    mma_scale_group<T, 10, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
    mma_scale_group<T, 12, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
    mma_scale_group<T, 14, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
    mma_scale_group<T, 0, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
    mma_scale_group<T, 2, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
    mma_scale_group<T, 4, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
    mma_scale_group<T, 6, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
    mma_scale_group<T, 8, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
    mma_scale_group<T, 10, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
    mma_scale_group<T, 12, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
    mma_scale_group<T, 14, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
    __builtin_amdgcn_s_setprio(0);

    // ===== Output writeback =====
    int epilogue_lane_id = lane_id;
    asm volatile("" : "+v"(epilogue_lane_id));
    auto p_coord_c = opus::make_tuple(wave_id_m, epilogue_lane_id % mma.grpn_c, wave_id_n, epilogue_lane_id / mma.grpn_c);
    auto u_gc = partition_layout_c<T::VEC_C>(mma, opus::make_tuple(c_lds_row_stride_elems, 1_I), p_coord_c);
    auto c_offset = [&](int half_tile_m, int half_tile_n) { return half_tile_m * T::HALF_B_M * c_lds_row_stride_elems + half_tile_n * T::HALF_B_N; };
    const int output_thread_id = wave_id * T::WARP_SIZE + epilogue_lane_id;

    // BF16 C store lds to global
    auto copy_output_bf16 = [&](int half_m, int half_n, int copy_index) {
        const int linear = output_thread_id * 8 + copy_index * T::BLOCK_SIZE * 8;
        const int output_row = linear / T::HALF_B_N + half_m * T::HALF_B_M;
        const int output_col = linear % T::HALF_B_N + half_n * T::HALF_B_N;
        const auto value = load<8>(s_c, output_row * c_lds_row_stride_elems + output_col);
        if (row + output_row < kargs.m) {
            store<8>(g_c, value, output_row * kargs.stride_c + output_col, 0, opus::number<2>{});
        }
    };
    const auto gc_offsets = opus::layout_to_offsets<T::VEC_C>(u_gc);

    // AGPR -> BF16 -> LDS
    auto stage_output_row = [&](auto index_i, int half_m, int half_n,
                                const AccFragment& c0, const AccFragment& c1,
                                const AccFragment& c2, const AccFragment& c3) {
        constexpr int index = decltype(index_i)::value;
        constexpr int n_repeat_stride = T::T_N * T::W_N;
        const int row_offset = gc_offsets[index] + c_offset(half_m, half_n);
        store<T::VEC_C>(s_c, cast<D_C>(c0), row_offset);
        store<T::VEC_C>(s_c, cast<D_C>(c1), row_offset + n_repeat_stride);
        store<T::VEC_C>(s_c, cast<D_C>(c2), row_offset + 2 * n_repeat_stride);
        store<T::VEC_C>(s_c, cast<D_C>(c3), row_offset + 3 * n_repeat_stride);
    };

    auto store_output_bf16 = [&](auto half_m_i, auto half_n_i) {
        constexpr int half_m = decltype(half_m_i)::value;
        constexpr int half_n = decltype(half_n_i)::value;
        __builtin_amdgcn_sched_barrier(0);
        if constexpr (!(half_m == 1 && half_n == 0)) {
            s_waitcnt_vmcnt(0_I);
        }
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
        opus::static_for<8>([&](auto copy_i) { copy_output_bf16(half_m, half_n, decltype(copy_i)::value); });
    };

    s_waitcnt_vmcnt(0_I);
    s_waitcnt_lgkmcnt(0_I);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    stage_output_row(opus::number<0>{}, 0, 0, c00[0], c00[1], c00[2], c00[3]);
    stage_output_row(opus::number<4>{}, 0, 0, c00[4], c00[5], c00[6], c00[7]);
    stage_output_row(opus::number<8>{}, 0, 0, c00[8], c00[9], c00[10], c00[11]);
    stage_output_row(opus::number<12>{}, 0, 0, c00[12], c00[13], c00[14], c00[15]);
    store_output_bf16(opus::number<0>{}, opus::number<0>{});

    stage_output_row(opus::number<0>{}, 1, 0, c10[0], c10[1], c10[2], c10[3]);
    stage_output_row(opus::number<4>{}, 1, 0, c10[4], c10[5], c10[6], c10[7]);
    stage_output_row(opus::number<8>{}, 1, 0, c10[8], c10[9], c10[10], c10[11]);
    stage_output_row(opus::number<12>{}, 1, 0, c10[12], c10[13], c10[14], c10[15]);
    store_output_bf16(opus::number<1>{}, opus::number<0>{});

    stage_output_row(opus::number<0>{}, 0, 1, c01[0], c01[1], c01[2], c01[3]);
    stage_output_row(opus::number<4>{}, 0, 1, c01[4], c01[5], c01[6], c01[7]);
    stage_output_row(opus::number<8>{}, 0, 1, c01[8], c01[9], c01[10], c01[11]);
    stage_output_row(opus::number<12>{}, 0, 1, c01[12], c01[13], c01[14], c01[15]);
    store_output_bf16(opus::number<0>{}, opus::number<1>{});

    stage_output_row(opus::number<0>{}, 1, 1, c11[0], c11[1], c11[2], c11[3]);
    stage_output_row(opus::number<4>{}, 1, 1, c11[4], c11[5], c11[6], c11[7]);
    stage_output_row(opus::number<8>{}, 1, 1, c11[8], c11[9], c11[10], c11[11]);
    stage_output_row(opus::number<12>{}, 1, 1, c11[12], c11[13], c11[14], c11[15]);
    store_output_bf16(opus::number<1>{}, opus::number<1>{});
}

#endif // AMDGPU hard-pin support
#endif // gfx950
