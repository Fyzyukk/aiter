    // Seed K0/K1 once. The complete scale panel has no runtime refill path.
    load_scale_panel();
    issue_matrix_prefetch(number<0>{}, number<0>{});
    issue_matrix_prefetch(number<1>{}, number<1>{});
    s_waitcnt_vmcnt(0_I);
    s_waitcnt_lgkmcnt(0_I);
    schedule_barrier();
    read_scales(number<0>{}, v_sfa, v_sfb);
    static_for<T::E_M>([&](auto m) { load_a(m, number<0>{}); });
    static_for<T::E_N>([&](auto n) { load_b(n, number<0>{}); });
    s_waitcnt_lgkmcnt(0_I);
    // Every wave must finish reading K0 before any producer reuses its slot.
    schedule_barrier();

    static_for<T::K_TILES - 1>([&](auto tile) {
        constexpr int group = decltype(tile)::value;
        constexpr int stage = group % T::NUM_STAGES;
        constexpr int next_stage = (group + 1) % T::NUM_STAGES;
        // Current operands are in registers; K+2 can replace their LDS slot.
        // Do not instantiate a prefetch beyond the final fixed-K group.
        if constexpr (group + 2 < T::K_TILES)
            issue_matrix_prefetch(number<stage>{}, number<group + 2>{});
        read_scales(number<group + 1>{}, v_sfa_next, v_sfb_next);
        // Replace each A after its last N use, and B after the last M use.
        static_for<T::E_M>([&](auto m) {
            static_for<T::E_N>([&](auto n) {
                compute(m, n);
                if constexpr (decltype(m)::value == T::E_M - 1)
                    load_b(n, number<next_stage>{});
            });
            load_a(m, number<next_stage>{});
        });
        // Publish the future slot and finish all reads before slot reuse.
        s_waitcnt_vmcnt(0_I);
        s_waitcnt_lgkmcnt(0_I);
        schedule_barrier();
        v_sfa = v_sfa_next;
        v_sfb = v_sfb_next;
    });
    static_for<T::E_M>([&](auto m) {
        static_for<T::E_N>([&](auto n) { compute(m, n); });
    });

