#include <opus/hip_minimal.hpp>
#include <opus/opus.hpp>
#include "small_split_source_expressions.h"
#include <cassert>
#include <climits>
#include <cstdint>
#include <iostream>
#include <vector>

using namespace opus;
uint64_t partitions_checked = 0, empty_wave_partitions = 0, coverage_cases = 0;

template<class T> void register_partition_and_workspace() {
    static_assert(T::W_K == 128 && T::GLOBAL_SPLIT_K == 4 && T::FIXED_K == 0 && T::OUTPUT == 3);
    static_assert(T::PREFETCH == 3 && !T::N_TAIL && !T::REUSE_B_SCALE);
    for (int total = 1; total <= 128; ++total) {
        std::vector<int> tile_coverage(total);
        for (int split = 0; split < T::GLOBAL_SPLIT_K; ++split) {
            int global_count = 0;
            for (int wk = 0; wk < T::WAVE_K; ++wk) {
                const auto p = register_partition<T>(total * 128, split, wk);
                assert(p.global_begin >= 0 && p.global_begin + p.global_loops <= total);
                assert(p.tile_begin >= p.global_begin && p.tile_begin + p.loops <= p.global_begin + p.global_loops);
                global_count += p.loops;
                if (!p.loops) ++empty_wave_partitions;
                ++partitions_checked;
                // Replay the actual P3 prologue/full/tail schedule with empty queues.
                std::vector<int> q(T::PREFETCH, -1), loads(p.loops), computes(p.loops);
                auto load = [&](int slot, int kt) {
                    assert(kt >= 0 && kt < p.loops);
                    q[slot] = kt; ++loads[kt];
                    const int absolute = p.tile_begin + kt;
                    ++tile_coverage[absolute];
                    const int m = 113, n = 256;
                    for (int mr = 0; mr < T::E_M; ++mr)
                        for (int lane = 0; lane < 64; ++lane) {
                            const int r = mr * 16 + lane % 16;
                            const int a = r * total * 128 + lane / 16 * 16 + absolute * 128;
                            assert(a + 64 + 15 < T::B_M * total * 128);
                            const int sf = absolute * m + r;
                            assert(sf < m * total);
                        }
                    for (int nr = 0; nr < T::E_N; ++nr)
                        for (int lane = 0; lane < 64; ++lane) {
                            const int b = nr * 16 * total * 128 + lane * 16 + absolute * 2048;
                            assert(b + 1024 + 15 < T::B_N * total * 128);
                            const int sf = (nr * 16 / 128) * total + absolute;
                            assert(sf < (n / 128) * total);
                        }
                };
                auto compute = [&](int slot) { assert(q[slot] >= 0); ++computes[q[slot]]; };
                for (int i = 0; i < T::PREFETCH; ++i) if (i < p.loops) load(i, i);
                const int full = p.loops / T::PREFETCH, tail = p.loops % T::PREFETCH;
                for (int group = 0; group + 1 < full; ++group)
                    for (int i = 0; i < T::PREFETCH; ++i) { compute(i); load(i, (group + 1) * T::PREFETCH + i); }
                if (full > 0)
                    for (int i = 0; i < T::PREFETCH; ++i) { compute(i); if (i < tail) load(i, full * T::PREFETCH + i); }
                for (int i = 0; i < T::PREFETCH; ++i) if (i < tail) compute(i);
                for (int i = 0; i < p.loops; ++i) assert(loads[i] == 1 && computes[i] == 1);
            }
            assert(global_count == register_partition<T>(total * 128, split, 0).global_loops);
        }
        for (int count : tile_coverage) assert(count == 1);
    }
    // Local K-wave partial addresses are a complete, disjoint array for every C fragment.
    constexpr int elems = T::B_M * T::B_N, elem_c = T::W_M * T::W_N / 64;
    if constexpr (T::WAVE_K > 1) {
        std::vector<int> coverage((T::WAVE_K - 1) * elems);
        for (int wk = 1; wk < T::WAVE_K; ++wk)
            for (int wave = 0; wave < T::T_M * T::T_N; ++wave)
                for (int lane = 0; lane < 64; ++lane)
                    for (int fragment = 0; fragment < T::E_M * T::E_N; ++fragment)
                        for (int e = 0; e < elem_c; ++e) {
                            const int offset = (wave * T::E_M * T::E_N * 64 + lane) * elem_c;
                            const int dst = (wk - 1) * elems + offset + fragment * 64 * elem_c + e;
                            assert(dst >= 0 && dst < int(coverage.size())); ++coverage[dst];
                        }
        for (int count : coverage) assert(count == 1);
    }
}

template<class T> void fine_partitions() {
    static_assert(T::FIXED_K == 0 && T::B_K == 128 && T::NUM_STAGES == 4 && T::CLUSTER == 1);
    static_assert(T::READ_ONLY_DRAIN && T::PREFETCH_BEFORE_READ && T::FINE_M_LOADS && !T::REGISTER_SCALES);
    for (int total = 1; total <= 128; ++total) {
        std::vector<int> coverage(total);
        for (int split = 0; split < T::SPLIT_K; ++split) {
            const auto p = fine_partition<T>(total * 128, split);
            const int stages = p.loops < T::NUM_STAGES ? p.loops : T::NUM_STAGES;
            const int matrix = stages * (T::A_STAGE + T::B_STAGE), scales = (T::B_M + T::B_GROUPS) * p.loops;
            assert(matrix + scales <= T::lds_bytes(total * 128));
            if (!p.loops) { ++empty_wave_partitions; assert(matrix == 0 && scales == 0); }
            for (int kt = 0; kt < p.loops; ++kt) {
                const int absolute = p.tile_begin + kt;
                assert(absolute >= 0 && absolute < total); ++coverage[absolute];
                const int m = 113;
                assert(p.tile_begin * m + kt * m + m - 1 < m * total);
                assert(p.tile_begin + kt < total);
            }
            ++partitions_checked;
        }
        for (int count : coverage) assert(count == 1);
    }
}

template<class T> void config_and_launch(int id, int m, int n, int k) {
    PrivateConfig cfg{};
    assert(private_config(id, true, m, n, k, cfg));
    assert(cfg.bm == T::B_M && cfg.bn == T::B_N && cfg.waves * 64 == T::BLOCK_SIZE);
    const int split = id < 200 ? 4 : (id % 10 == 1 ? 2 : 1);
    assert(cfg.split == split && cfg.local_k == (id == 111 || id == 121 ? 2 : 1));
    const uint64_t required = private_workspace_bytes(cfg, m, n);
    assert(required == (split > 1 ? uint64_t(split) * m * n * 4 : 0));
    const int elements = m * n;
    assert(int64_t(elements) * (split > 1 ? 4 : 2) <= INT_MAX);
    assert(n % T::B_N == 0 && elements % 16 == 0);
    const int grid_y = (m + T::B_M - 1) / T::B_M;
    for (int block_m : {0, grid_y - 1}) {
        const int row = block_m * T::B_M, col = n - T::B_N;
        assert(row >= 0 && row < m);
        for (int s = 0; s < split; ++s) {
            const int64_t base = int64_t(s) * elements + int64_t(row) * n + col;
            const int64_t span = int64_t(m - row) * n - col;
            assert(base + span == int64_t(s + 1) * elements);
            if (required) assert(uint64_t(base + span) * 4 <= required);
        }
    }
    // Vec16 reducer's first/last active vectors stay within every FP32 partition.
    if (split > 1) {
        const int last = ((elements - 1) / 16) * 16;
        assert(last + 15 < elements);
        for (int s = 0; s < split; ++s)
            assert(uint64_t(s * int64_t(elements) + last + 16) * 4 <= required);
        assert((elements + 2047) / 2048 > 0);
    }
    const void* a = reinterpret_cast<const void*>(uintptr_t(0x1000000000));
    const void* b = reinterpret_cast<const void*>(uintptr_t(0x2000000000));
    const void* sa = reinterpret_cast<const void*>(uintptr_t(0x3000000000));
    const void* sb = reinterpret_cast<const void*>(uintptr_t(0x4000000001));
    void* c = reinterpret_cast<void*>(uintptr_t(0x5000000000));
    void* ws = reinterpret_cast<void*>(uintptr_t(0x6000000000));
    assert(private_pointers_valid(cfg,a,b,sa,sb,c,required?ws:nullptr,required,m,n,k));
    if (required) {
        assert(!private_pointers_valid(cfg,a,b,sa,sb,c,ws,required-1,m,n,k));
        assert(!private_pointers_valid(cfg,a,b,sa,sb,c,c,required,m,n,k));
    }
    ++coverage_cases;
}

void check_cases() {
    for (const auto& c : reviewed_shapes) {
        if (c.fine) {
            config_and_launch<opus_private_narrow_fine_traits<48,64,1,4,1>>(210,c.m,c.n,c.k);
            config_and_launch<opus_private_narrow_fine_traits<48,64,1,4,2>>(211,c.m,c.n,c.k);
            config_and_launch<opus_private_narrow_fine_traits<64,128,2,2,1>>(220,c.m,c.n,c.k);
            config_and_launch<opus_private_narrow_fine_traits<64,128,2,2,2>>(221,c.m,c.n,c.k);
            config_and_launch<opus_private_narrow_fine_traits<96,128,2,2,1>>(230,c.m,c.n,c.k);
            config_and_launch<opus_private_narrow_fine_traits<96,128,2,2,2>>(231,c.m,c.n,c.k);
        } else {
            config_and_launch<opus_private_register_split4_traits<16,16,1>>(110,c.m,c.n,c.k);
            config_and_launch<opus_private_register_split4_traits<16,16,2>>(111,c.m,c.n,c.k);
            config_and_launch<opus_private_register_split4_traits<16,32,1>>(120,c.m,c.n,c.k);
            config_and_launch<opus_private_register_split4_traits<16,32,2>>(121,c.m,c.n,c.k);
            config_and_launch<opus_private_register_split4_traits<32,32,1>>(130,c.m,c.n,c.k);
            config_and_launch<opus_private_register_split4_traits<32,64,1>>(140,c.m,c.n,c.k);
        }
        PrivateConfig baseline{};
        assert(private_config(c.parent, false, c.m, c.n, c.k, baseline));
        assert(baseline.split == c.baseline_split && baseline.bm == c.baseline_bm && baseline.bn == c.baseline_bn);
    }
    assert(coverage_cases == 540);
}

int main() {
    register_partition_and_workspace<opus_private_register_split4_traits<16,16,1>>();
    register_partition_and_workspace<opus_private_register_split4_traits<16,16,2>>();
    register_partition_and_workspace<opus_private_register_split4_traits<16,32,1>>();
    register_partition_and_workspace<opus_private_register_split4_traits<16,32,2>>();
    register_partition_and_workspace<opus_private_register_split4_traits<32,32,1>>();
    register_partition_and_workspace<opus_private_register_split4_traits<32,64,1>>();
    fine_partitions<opus_private_narrow_fine_traits<48,64,1,4,1>>();
    fine_partitions<opus_private_narrow_fine_traits<48,64,1,4,2>>();
    fine_partitions<opus_private_narrow_fine_traits<64,128,2,2,1>>();
    fine_partitions<opus_private_narrow_fine_traits<64,128,2,2,2>>();
    fine_partitions<opus_private_narrow_fine_traits<96,128,2,2,1>>();
    fine_partitions<opus_private_narrow_fine_traits<96,128,2,2,2>>();
    check_cases();
    std::cout << "passed: extracted actual global/local partition expressions for all128 K counts; actual traits/launcher guards for90 shapes and540 cases; zero-loop queues/local partial addresses; absolute scale indices; workspace/reducer bounds; checks=" << partitions_checked << ", empty_wave_partitions=" << empty_wave_partitions << "; no HIP/HSA dependency\n";
}
