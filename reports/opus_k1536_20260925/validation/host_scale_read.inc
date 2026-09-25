    auto read_scales = [&](auto tile, auto& scale_a, auto& scale_b) {
        const auto group = tile;
        if constexpr (is_constant_v<decltype(tile)>)
            static_assert(decltype(tile)::value >= 0 && decltype(tile)::value < T::K_TILES);
        static_for<T::A_SCALE_PACKS>([&](auto p) { scale_a[decltype(p)::value] = 0; });
        static_for<T::E_M>([&](auto i) {
            constexpr int repeat = decltype(i)::value;
            const int local_row = wave_id_m * T::W_M + lane_id % T::W_M + repeat * T::T_M * T::W_M;
            const unsigned value = cpu_check::load<1>(s_sfa, group * number<T::B_M>{} + local_row)[0];
            scale_a[repeat / 4] |= value << ((repeat % 4) * 8);
        });
        static_for<T::B_SCALE_PACKS>([&](auto i) {
            constexpr int half_n = decltype(i)::value;
            scale_b[half_n] = cpu_check::load<1>(s_sfb, half_n * T::SCALE_PANEL + group)[0];
        });
    };
