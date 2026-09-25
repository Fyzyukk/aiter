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

    auto advance = [&](auto stage, auto tile, auto has_prefetch) {
        constexpr int slot = decltype(stage)::value;
        constexpr int next_stage = (slot + 1) % T::NUM_STAGES;
        // Current operands are in registers; K+2 can replace their LDS slot.
        // Stage stays static even when the pair loop supplies a runtime tile.
        if constexpr (decltype(has_prefetch)::value)
            issue_matrix_prefetch(stage, tile + number<2>{});
        read_scales(tile + number<1>{}, v_sfa_next, v_sfb_next);
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
    };

    if constexpr (T::LOOP_UNROLL == 12) {
        // Eleven static advances followed by the final static K11 compute.
        static_for<T::K_TILES - 1>([&](auto tile) {
            constexpr int group = decltype(tile)::value;
            constexpr int stage = group % T::NUM_STAGES;
            advance(number<stage>{}, tile, bool_constant<(group + 2 < T::K_TILES)>{});
        });
    } else {
        // Five dynamic pairs compute K0..K9, each step with its own barrier.
        // Only the K index varies at runtime; each LDS slot is compile-time.
#pragma clang loop unroll(disable)
        for (int group = 0; group < 10; group += 2) {
            advance(number<0>{}, group, true_type{});
            advance(number<1>{}, group + 1, true_type{});
        }
        // K11 is already resident: advance K10 without a K12 prefetch.
        advance(number<0>{}, number<10>{}, false_type{});
    }
    // Both paths have 2 prologue + 11 advance barriers, then consume K11.
    static_for<T::E_M>([&](auto m) {
        static_for<T::E_N>([&](auto n) { compute(m, n); });
    });

