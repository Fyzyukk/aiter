    auto compute = [&](auto m, auto n) {
        constexpr int mi = decltype(m)::value, ni = decltype(n)::value;
        constexpr int ci = mi * T::E_N + ni;
        constexpr int half_n = ni / (T::GROUP_N / (T::T_N * T::W_N));
        c00[ci] = BaseMMA{}(v_a[mi], v_b[ni], c00[ci],
                           static_cast<int>(v_sfa[mi / 4]), static_cast<int>(v_sfb[half_n]),
                           number<mi % 4>{}, number<0>{});
    };

