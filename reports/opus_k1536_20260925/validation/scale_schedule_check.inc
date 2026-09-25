template<int LoopUnroll> void check_scales() {
    using namespace opus;
    using T = opus_gemm_mxscale_bpreshuffle_k1536_traits_gfx950<LoopUnroll>;
    using D_SF = unsigned char;
    std::uint64_t cases = 0, scale_values_checked = 0;
    for (int valid_rows : valid_m_rows) for (int row : {0, 192})
        for (int n : n_values) for (int col : {0, n - T::B_N})
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
                        if constexpr (LoopUnroll == 2 && group > 0 && group < 11)
                            read_scales(int(group), scale_a, scale_b);
                        else
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
    std::cout << "K=" << T::FIXED_K << " LoopUnroll=" << LoopUnroll << " scale_cases=" << cases << " scale_values_checked=" << scale_values_checked
              << " exact_panel_bytes=" << T::SFA_BYTES + T::SFB_BYTES << '\n';
}

namespace cpu_check {
using Trace = std::vector<std::array<int, 5>>;
enum Event { Panel = 1, Prefetch, WaitVM, WaitLDS, Barrier, Scale, ReadA, ReadB, MFMA };
struct Operand { int group = -1, index = -1; bool ready = false; };
struct Accumulator { int index = -1, groups = 0; };
template<class T> struct ScheduleContext {
    std::array<std::array<int, T::E_M * T::E_N>, T::K_TILES> computes{};
    std::array<bool, T::K_TILES> scale_read{}, scale_ready{};
    int barrier_count = 0, mfma_calls = 0;
    Trace trace;
    void event(int kind, int group = -1, int x = -1, int y = -1) {
        trace.push_back({kind, group, x, y, barrier_count});
    }
};
template<class T> struct InstrumentedMMA {
    inline static ScheduleContext<T>* active = nullptr;
    template<class ScaleA, class ScaleB>
    Accumulator operator()(const Operand& a, const Operand& b, Accumulator c,
                           int packed_a, int packed_b, ScaleA, ScaleB) const {
        static_assert(ScaleA::value >= 0 && ScaleA::value < 4 && ScaleB::value == 0);
        auto& context = *active;
        const int group = a.group;
        assert(group >= 0 && group < T::K_TILES && b.group == group);
        assert(a.index >= 0 && a.index < T::E_M && b.index >= 0 && b.index < T::E_N);
        assert(a.ready && b.ready && context.scale_ready[group]);
        assert(ScaleA::value == a.index % 4);
        assert(((static_cast<unsigned>(packed_a) >> (8 * ScaleA::value)) & 255) ==
               static_cast<unsigned>((group + 1) * 4 + a.index));
        assert(packed_b == (group + 1) * 2 + b.index / 4);
        const int index = a.index * T::E_N + b.index;
        assert(c.index == index && c.groups == group);
        assert(++context.computes[group][index] == 1);
        ++context.mfma_calls;
        context.event(MFMA, group, a.index, b.index);
        ++c.groups;
        return c;
    }
};
}

template<int LoopUnroll> cpu_check::Trace check_schedule() {
    using namespace opus;
    using T = opus_gemm_mxscale_bpreshuffle_k1536_traits_gfx950<LoopUnroll>;
    using BaseMMA = cpu_check::InstrumentedMMA<T>;
    cpu_check::ScheduleContext<T> context;
    BaseMMA::active = &context;
    struct Slot { int tile = -1; bool pending = false; int last_read_barrier = -1; int a_reads = 0, b_reads = 0; };
    std::array<Slot, T::NUM_STAGES> slots;
    std::array<int, T::K_TILES> prefetch_count{};
    std::array<cpu_check::Operand, T::E_M> v_a;
    std::array<cpu_check::Operand, T::E_N> v_b;
    std::array<cpu_check::Accumulator, T::E_M * T::E_N> c00;
    for (int i = 0; i < static_cast<int>(c00.size()); ++i) c00[i].index = i;
    std::array<unsigned, T::A_SCALE_PACKS> v_sfa{}, v_sfa_next{};
    std::array<unsigned, T::B_SCALE_PACKS> v_sfb{}, v_sfb_next{};
    int scale_panel_loads = 0, last_scale_read = -1, runtime_prefetches = 0, runtime_scale_reads = 0;
    bool vm_waited = false, lds_waited = false, scales_published = false;
    auto load_scale_panel = [&]() {
        assert(++scale_panel_loads == 1);
        vm_waited = lds_waited = false;
        context.event(cpu_check::Panel);
    };
    auto issue_matrix_prefetch = [&](auto stage, auto tile) {
        constexpr int slot = decltype(stage)::value;
        const int group = static_cast<int>(tile);
        static_assert(slot >= 0 && slot < T::NUM_STAGES);
        assert(group >= 0 && group < T::K_TILES);
        assert(slot == group % T::NUM_STAGES);
        if constexpr (!is_constant_v<decltype(tile)>) ++runtime_prefetches;
        auto& state = slots[slot];
        if (state.tile >= 0) {
            assert(!state.pending);
            assert(state.a_reads == T::E_M && state.b_reads == T::E_N);
            assert(state.last_read_barrier < context.barrier_count);
        }
        assert(++prefetch_count[group] == 1);
        state = Slot{group, true, -1, 0, 0};
        vm_waited = lds_waited = false;
        context.event(cpu_check::Prefetch, group, slot);
    };
    auto s_waitcnt_vmcnt = [&](auto value) {
        static_assert(decltype(value)::value == 0);
        vm_waited = true;
        context.event(cpu_check::WaitVM);
    };
    auto s_waitcnt_lgkmcnt = [&](auto value) {
        static_assert(decltype(value)::value == 0);
        lds_waited = true;
        for (auto& a : v_a) if (a.group >= 0) a.ready = true;
        for (auto& b : v_b) if (b.group >= 0) b.ready = true;
        for (int group = 0; group < T::K_TILES; ++group)
            if (context.scale_read[group]) context.scale_ready[group] = true;
        context.event(cpu_check::WaitLDS);
    };
    auto schedule_barrier = [&]() {
        assert(lds_waited);
        for (auto& slot : slots) if (slot.pending) { assert(vm_waited); slot.pending = false; }
        if (scale_panel_loads && !scales_published) { assert(vm_waited); scales_published = true; }
        ++context.barrier_count;
        context.event(cpu_check::Barrier);
    };
    auto read_scales = [&](auto tile, auto& a, auto& b) {
        const int group = static_cast<int>(tile);
        assert(group >= 0 && group < T::K_TILES);
        assert(scales_published && group == ++last_scale_read);
        if constexpr (!is_constant_v<decltype(tile)>) ++runtime_scale_reads;
        a.fill(0);
        for (int mi = 0; mi < T::E_M; ++mi)
            a[mi / 4] |= static_cast<unsigned>((group + 1) * 4 + mi) << (8 * (mi % 4));
        for (int half = 0; half < T::B_SCALE_PACKS; ++half) b[half] = (group + 1) * 2 + half;
        context.scale_read[group] = true;
        lds_waited = false;
        context.event(cpu_check::Scale, group);
    };
    auto load_a = [&](auto m, auto stage) {
        constexpr int mi = decltype(m)::value, slot = decltype(stage)::value;
        auto& state = slots[slot];
        assert(state.tile >= 0 && !state.pending);
        if (v_a[mi].group >= 0)
            for (int ni = 0; ni < T::E_N; ++ni)
                assert(context.computes[v_a[mi].group][mi * T::E_N + ni] == 1);
        v_a[mi] = {state.tile, mi, false};
        state.last_read_barrier = context.barrier_count;
        ++state.a_reads;
        lds_waited = false;
        context.event(cpu_check::ReadA, state.tile, mi, slot);
    };
    auto load_b = [&](auto n, auto stage) {
        constexpr int ni = decltype(n)::value, slot = decltype(stage)::value;
        auto& state = slots[slot];
        assert(state.tile >= 0 && !state.pending);
        if (v_b[ni].group >= 0)
            for (int mi = 0; mi < T::E_M; ++mi)
                assert(context.computes[v_b[ni].group][mi * T::E_N + ni] == 1);
        v_b[ni] = {state.tile, ni, false};
        state.last_read_barrier = context.barrier_count;
        ++state.b_reads;
        lds_waited = false;
        context.event(cpu_check::ReadB, state.tile, ni, slot);
    };
    // The actual production compute lambda selects registers, scale bytes,
    // scale halves, and accumulator indices for the CPU MFMA oracle.
#include "host_compute.inc"
    // Exact production schedule, with only the barrier intrinsic redirected.
#include "host_schedule.inc"
    for (int count : prefetch_count) assert(count == 1);
    for (const auto& group : context.computes) for (int count : group) assert(count == 1);
    for (const auto& c : c00) assert(c.groups == T::K_TILES);
    for (auto& state : slots) assert(!state.pending && state.a_reads == T::E_M && state.b_reads == T::E_N);
    assert(scale_panel_loads == 1 && last_scale_read == T::K_TILES - 1);
    assert(context.barrier_count == 13 && context.mfma_calls == 288);
    assert(runtime_prefetches == (LoopUnroll == 2 ? 10 : 0));
    assert(runtime_scale_reads == (LoopUnroll == 2 ? 10 : 0));
    std::cout << "K=" << T::FIXED_K << " LoopUnroll=" << LoopUnroll << " fixed_groups=" << T::K_TILES
              << " matrix_prefetches=" << T::K_TILES << " MFMA_calls_per_wave=" << context.mfma_calls
              << " barriers=" << context.barrier_count << " scale_panel_loads=" << scale_panel_loads
              << " runtime_prefetches=" << runtime_prefetches << " runtime_scale_reads=" << runtime_scale_reads
              << '\n';
    BaseMMA::active = nullptr;
    return context.trace;
}
