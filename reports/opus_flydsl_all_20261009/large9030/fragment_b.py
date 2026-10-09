"""Reduce the private direct-B queue from full tiles to two MFMA fragments."""
def fragment_b(source):
    source = source.replace('typename decltype(mma)::vtype_b v_b, v_b_next;', 'vector_t<D_B, ELEM_B> v_b, v_b_next;')
    start = source.index('    // Two register sets:')
    stop = source.index('    auto mma_scale_fragment =', start)
    source = source[:start] + '''    // Two single-fragment register sets; the N queue carries across K tiles.
    auto load_b_direct = [&](auto& target, int tile_k, auto n_i) {
        constexpr int n_repeat = decltype(n_i)::value;
        const int group = n_repeat * T::T_N + wave_id_n;
        const int base = group * T::W_N * kargs.stride_b + tile_k * T::B_K * T::W_N + lane_id * T::VEC_B;
        static_for<T::B_CHUNKS_PER_FRAGMENT>([&](auto chunk_i) {
            constexpr int chunk = decltype(chunk_i)::value;
            set_slice(target, load<T::VEC_B>(g_b, base + chunk * T::WARP_SIZE * T::VEC_B),
                      number<chunk * T::VEC_B>{}, number<(chunk + 1) * T::VEC_B>{});
        });
    };
''' + source[stop:]
    source = source.replace('const auto b = slice(v_b, number<n_repeat * ELEM_B>{}, number<(n_repeat + 1) * ELEM_B>{});', 'const auto b = v_b;')
    start = source.index('    // Fill two of three A slots;')
    stop = source.index('    // All A readers finish', start)
    source = source[:start] + '''    // Prefetch one B fragment while all M repeats consume the current fragment.
    auto compute_tile = [&](int tile_k) {
        static_for<T::E_N>([&](auto n_i) {
            constexpr int n_repeat = decltype(n_i)::value;
            static_for<T::E_M>([&](auto m_i) { mma_scale_fragment(m_i, n_i); });
            if constexpr (n_repeat + 1 < T::E_N) {
                s_waitcnt_vmcnt(0_I);
                v_b = v_b_next;
                if constexpr (n_repeat + 2 < T::E_N)
                    load_b_direct(v_b_next, tile_k, number<n_repeat + 2>{});
                else if (tile_k + 1 < loops)
                    load_b_direct(v_b_next, tile_k + 1, number<0>{});
            } else if (tile_k + 1 < loops) {
                s_waitcnt_vmcnt(0_I);
                v_b = v_b_next;
                load_b_direct(v_b_next, tile_k + 1, number<1>{});
            }
        });
    };
    // Fill two of three A slots; the third is free while current A is consumed.
    load_sfa_panel(0);
    load_sfb_panel(0);
    issue_matrix_prefetch(0, 0);
    issue_matrix_prefetch(1, 1);
    load_b_direct(v_b, 0, number<0>{});
    load_b_direct(v_b_next, 0, number<1>{});
    s_waitcnt_vmcnt(0_I);
    s_waitcnt_lgkmcnt(0_I);
    __builtin_amdgcn_s_barrier();
    read_scales(0, v_sfa, v_sfb);
    static_for<T::E_M>([&](auto m_i) { load_a_fragment(m_i, 0); });
    s_waitcnt_lgkmcnt(0_I);
    __builtin_amdgcn_s_barrier();

    // Three-slot A ring, two B fragments. Keep completion and reader retirement.
    #pragma unroll 1
    for (int tile_k = 0; tile_k + 1 < loops; ++tile_k) {
        if (tile_k + 2 < loops)
            issue_matrix_prefetch((tile_k + 2) % T::NUM_STAGES, tile_k + 2);
        compute_tile(tile_k);
        // Completes future A and next B fragment; barrier publishes A copies.
        s_waitcnt_vmcnt(0_I);
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_barrier();
        read_scales(tile_k + 1, v_sfa_next, v_sfb_next);
        static_for<T::E_M>([&](auto m_i) { load_a_fragment(m_i, (tile_k + 1) % T::NUM_STAGES); });
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_barrier();
        v_sfa = v_sfa_next;
        v_sfb = v_sfb_next;
    }
    compute_tile(loops - 1);
''' + source[stop:]
    return source
