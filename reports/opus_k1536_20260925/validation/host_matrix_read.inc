    auto load_a = [&](auto m, auto stage) {
        constexpr int repeat = decltype(m)::value;
        constexpr int slot = decltype(stage)::value;
        static_for<2>([&](auto c) {
            constexpr int index = repeat * 2 + decltype(c)::value;
            const int offset = T::a_lds_offset(wave_id_m, lane_id, repeat, decltype(c)::value) + slot * T::A_STAGE;
            a_chunks[index] = cpu_check::load<16>(s_a, offset);
        });
    };
    auto load_b = [&](auto n, auto stage) {
        constexpr int repeat = decltype(n)::value;
        constexpr int slot = decltype(stage)::value;
        const auto offsets = layout_to_offsets<16>(u_rb + slot * T::B_STAGE);
        static_for<2>([&](auto c) {
            constexpr int index = repeat * 2 + decltype(c)::value;
            b_chunks[index] = cpu_check::load<16>(s_b, offsets[index]);
        });
    };
