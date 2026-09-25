namespace cpu_check {
struct Memory {
    std::vector<unsigned char> values;
    std::vector<int> reads, writes;
    explicit Memory(int size) : values(size, 0xa5), reads(size, 0), writes(size, 0) {}
};
template<int Vec> auto load(Memory& memory, int index) {
    assert(index >= 0 && index + Vec <= static_cast<int>(memory.values.size()));
    opus::vector_t<unsigned char, Vec> result;
    for (int j = 0; j < Vec; ++j) {
        ++memory.reads[index + j];
        result[j] = memory.values[index + j];
    }
    return result;
}
template<int Vec, class Value> void store(Memory& memory, const Value& value, int index) {
    assert(index >= 0 && index + Vec <= static_cast<int>(memory.values.size()));
    for (int j = 0; j < Vec; ++j) {
        assert(++memory.writes[index + j] == 1);
        memory.values[index + j] = value[j];
    }
}
}

template<int FixedK> void check_scales() {
    using namespace opus;
    using T = opus_gemm_mxscale_bpreshuffle_shortk_traits_gfx950<FixedK>;
    using D_SF = unsigned char;
    std::uint64_t cases = 0, scale_values_checked = 0;
    for (int valid_rows : {64, 128, 192}) for (int row : {0, 192})
        for (int n : {256, 6144, 7168}) for (int col : {0, n - T::B_N})
            for (int shift : {0, 8}) {
                struct { int m, stride_sfa, stride_sfb; } kargs{row + valid_rows, row + valid_rows, T::K_TILES};
                cpu_check::Memory g_sfa(kargs.m * T::K_TILES), g_sfb(n / T::GROUP_N * T::K_TILES);
                cpu_check::Memory s_sfa(T::SFA_BYTES), s_sfb(T::SFB_BYTES);
                for (int i = 0; i < static_cast<int>(g_sfa.values.size()); ++i) g_sfa.values[i] = (i >> shift) & 255;
                for (int i = 0; i < static_cast<int>(g_sfb.values.size()); ++i) g_sfb.values[i] = (i >> shift) & 255;
                // Execute the production scale producer body for every thread.
                for (int cpu_thread = 0; cpu_thread < T::BLOCK_SIZE; ++cpu_thread) {
#include "host_scale_load.inc"
                    load_scale_panel();
                }
                for (int group = 0; group < T::K_TILES; ++group) {
                    for (int local_row = 0; local_row < T::B_M; ++local_row) {
                        const int dst = group * T::B_M + local_row;
                        const int src = group * kargs.stride_sfa + row + local_row;
                        const unsigned expected = local_row < valid_rows ? g_sfa.values[src] : 0x7f;
                        assert(s_sfa.writes[dst] == 1 && s_sfa.values[dst] == expected);
                    }
                    for (int half_n = 0; half_n < 2; ++half_n) {
                        const int dst = half_n * T::SCALE_PANEL + group;
                        const int src = (col / T::GROUP_N + half_n) * kargs.stride_sfb + group;
                        assert(s_sfb.writes[dst] == 1 && s_sfb.values[dst] == g_sfb.values[src]);
                    }
                }
                for (int i = 0; i < static_cast<int>(g_sfa.reads.size()); ++i)
                    assert(g_sfa.reads[i] == (i % kargs.m >= row && i % kargs.m < row + valid_rows));
                for (int i = 0; i < static_cast<int>(g_sfb.reads.size()); ++i)
                    assert(g_sfb.reads[i] == (i / T::K_TILES >= col / T::GROUP_N && i / T::K_TILES < col / T::GROUP_N + 2));
                // Execute the production scale consumer for all lanes/groups.
                for (int cpu_thread = 0; cpu_thread < T::BLOCK_SIZE; ++cpu_thread) {
                    const int lane_id = cpu_thread % 64, wave_id_m = (cpu_thread / 64) % T::T_M;
#include "host_scale_read.inc"
                    static_for<T::K_TILES>([&](auto tile) {
                        constexpr int group = decltype(tile)::value;
                        std::array<unsigned, T::A_SCALE_PACKS> scale_a;
                        std::array<unsigned, T::B_SCALE_PACKS> scale_b;
                        scale_a.fill(0xffffffffu);
                        scale_b.fill(0xffffffffu);
                        read_scales(tile, scale_a, scale_b);
                        for (int mi = 0; mi < T::E_M; ++mi) {
                            const int local_row = mi * T::T_M * 16 + wave_id_m * 16 + lane_id % 16;
                            const int src = group * kargs.stride_sfa + row + local_row;
                            const unsigned expected = local_row < valid_rows ? g_sfa.values[src] : 0x7f;
                            assert(((scale_a[mi / 4] >> (8 * (mi % 4))) & 255) == expected);
                            ++scale_values_checked;
                        }
                        assert((scale_a.back() >> 24) == 0);
                        for (int half_n = 0; half_n < 2; ++half_n) {
                            const int src = (col / T::GROUP_N + half_n) * kargs.stride_sfb + group;
                            assert(scale_b[half_n] == g_sfb.values[src]);
                            ++scale_values_checked;
                        }
                    });
                }
                for (int reads : s_sfa.reads) assert(reads == 8);
                for (int reads : s_sfb.reads) assert(reads == T::BLOCK_SIZE);
                ++cases;
            }
    std::cout << "K=" << FixedK << " scale_cases=" << cases << " scale_values_checked=" << scale_values_checked
              << " exact_panel_bytes=" << T::SFA_BYTES + T::SFB_BYTES << '\n';
}

template<int FixedK> void check_schedule() {
    using namespace opus;
    using T = opus_gemm_mxscale_bpreshuffle_shortk_traits_gfx950<FixedK>;
    struct Slot { int tile = -1; bool pending = false; int last_read_barrier = -1; int a_reads = 0, b_reads = 0; };
    std::array<Slot, T::NUM_STAGES> slots;
    std::array<int, T::K_TILES> prefetch_count{};
    std::array<std::array<int, T::E_M * T::E_N>, T::K_TILES> computes{};
    std::array<int, T::E_M> a_groups;
    std::array<int, T::E_N> b_groups;
    a_groups.fill(-1);
    b_groups.fill(-1);
    std::array<int, T::A_SCALE_PACKS> v_sfa{}, v_sfa_next{};
    std::array<int, T::B_SCALE_PACKS> v_sfb{}, v_sfb_next{};
    int barrier_count = 0, scale_panel_loads = 0, last_scale_read = -1;
    bool vm_waited = false, lds_waited = false, scales_published = false;
    auto load_scale_panel = [&]() { assert(++scale_panel_loads == 1); vm_waited = lds_waited = false; };
    auto issue_matrix_prefetch = [&](auto stage, auto tile) {
        constexpr int slot = decltype(stage)::value, group = decltype(tile)::value;
        static_assert(slot >= 0 && slot < T::NUM_STAGES && group >= 0 && group < T::K_TILES);
        auto& state = slots[slot];
        if (state.tile >= 0) {
            assert(!state.pending);
            assert(state.a_reads == T::E_M && state.b_reads == T::E_N);
            assert(state.last_read_barrier < barrier_count);
        }
        assert(++prefetch_count[group] == 1);
        state = Slot{group, true, -1, 0, 0};
        vm_waited = lds_waited = false;
    };
    auto s_waitcnt_vmcnt = [&](auto value) { static_assert(decltype(value)::value == 0); vm_waited = true; };
    auto s_waitcnt_lgkmcnt = [&](auto value) { static_assert(decltype(value)::value == 0); lds_waited = true; };
    auto schedule_barrier = [&]() {
        assert(lds_waited);
        for (auto& slot : slots) if (slot.pending) { assert(vm_waited); slot.pending = false; }
        if (scale_panel_loads && !scales_published) { assert(vm_waited); scales_published = true; }
        ++barrier_count;
    };
    auto read_scales = [&](auto tile, auto& a, auto& b) {
        constexpr int group = decltype(tile)::value;
        static_assert(group >= 0 && group < T::K_TILES);
        assert(scales_published && group == ++last_scale_read);
        a.fill(group);
        b.fill(group);
        lds_waited = false;
    };
    auto load_a = [&](auto m, auto stage) {
        constexpr int mi = decltype(m)::value, slot = decltype(stage)::value;
        auto& state = slots[slot];
        assert(state.tile >= 0 && !state.pending);
        if (a_groups[mi] >= 0)
            for (int ni = 0; ni < T::E_N; ++ni) assert(computes[a_groups[mi]][mi * T::E_N + ni] == 1);
        a_groups[mi] = state.tile;
        state.last_read_barrier = barrier_count;
        ++state.a_reads;
        lds_waited = false;
    };
    auto load_b = [&](auto n, auto stage) {
        constexpr int ni = decltype(n)::value, slot = decltype(stage)::value;
        auto& state = slots[slot];
        assert(state.tile >= 0 && !state.pending);
        if (b_groups[ni] >= 0)
            for (int mi = 0; mi < T::E_M; ++mi) assert(computes[b_groups[ni]][mi * T::E_N + ni] == 1);
        b_groups[ni] = state.tile;
        state.last_read_barrier = barrier_count;
        ++state.b_reads;
        lds_waited = false;
    };
    auto compute = [&](auto m, auto n) {
        constexpr int mi = decltype(m)::value, ni = decltype(n)::value;
        const int group = a_groups[mi];
        assert(group >= 0 && b_groups[ni] == group);
        assert(v_sfa[mi / 4] == group && v_sfb[ni / 4] == group);
        assert(++computes[group][mi * T::E_N + ni] == 1);
    };
    // Exact production schedule, with only the barrier intrinsic redirected.
#include "host_schedule.inc"
    for (int count : prefetch_count) assert(count == 1);
    for (const auto& group : computes) for (int count : group) assert(count == 1);
    for (auto& state : slots) assert(!state.pending && state.a_reads == T::E_M && state.b_reads == T::E_N);
    assert(scale_panel_loads == 1 && last_scale_read == T::K_TILES - 1);
    assert(barrier_count == T::K_TILES + 1);
    std::cout << "K=" << FixedK << " fixed_groups=" << T::K_TILES
              << " matrix_prefetches=" << T::K_TILES
              << " MFMA_calls_per_wave=" << T::K_TILES * T::E_M * T::E_N
              << " barriers=" << barrier_count << " scale_panel_loads=" << scale_panel_loads << '\n';
}
