    auto issue_matrix_prefetch = [&](auto stage, auto tile) {
        constexpr int slot = decltype(stage)::value;
        // Preserve number<> offsets in the fully expanded specialization.
        const auto group = tile;
        static_assert(slot >= 0 && slot < T::NUM_STAGES);
        if constexpr (is_constant_v<decltype(tile)>)
            static_assert(decltype(tile)::value >= 0 && decltype(tile)::value < T::K_TILES);
        cpu_check::async_load<16>(g_a, s_a.ptr, u_ga, u_sa + slot * T::A_STAGE, group * number<T::B_K>{});
        cpu_check::async_load<16>(g_b, s_b.ptr, u_gb, u_sb + slot * T::B_STAGE,
                       group * number<T::B_K * T::W_N>{});
    };
