    auto load_scale_panel = [&]() {
        static_for<T::SFA_PASSES>([&](auto i) {
            const int index = (cpu_thread + decltype(i)::value * T::BLOCK_SIZE) * 16;
            const int group = index / T::B_M;
            const int local_row = index % T::B_M;
            if (index < T::SFA_BYTES) {
                vector_t<D_SF, 16> values;
                if (row + local_row < kargs.m)
                    values = cpu_check::load<16>(g_sfa, group * kargs.stride_sfa + row + local_row);
                else
                    static_for<16>([&](auto j) { values[decltype(j)::value] = 0x7f; });
                cpu_check::store<16>(s_sfa, values, index);
            }
        });
        if (cpu_thread < T::SFB_BYTES) {
            const int half_n = cpu_thread / T::SCALE_PANEL;
            const int group = cpu_thread % T::SCALE_PANEL;
            cpu_check::store<1>(s_sfb, cpu_check::load<1>(g_sfb, (col / T::GROUP_N + half_n) * kargs.stride_sfb + group),
                     cpu_thread);
        }
    };
