// CPU-only coordinate validation; no HIP runtime functions are called.
#include <hip/hip_runtime.h>
#include <opus/opus.hpp>
#include "opus_gemm_traits_a8w8_mxscale_bpreshuffle_k1536_gfx950.cuh"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>
using opus::operator""_I;
#include "host_layout_helpers.inc"
#include "cpu_memory.inc"

constexpr std::array<int, 6> n_values{256, 512, 6144, 7168, 8192, 16384};
constexpr std::array<int, 3> valid_m_rows{64, 128, 192};

template<int LoopUnroll> void check_matrices_and_output() {
    using namespace opus;
    using T = opus_gemm_mxscale_bpreshuffle_k1536_traits_gfx950<LoopUnroll>;
    static_assert(T::FIXED_K == 1536 && T::K_TILES == 12 && T::LOOP_UNROLL == LoopUnroll);
    static_assert(T::B_M == 192 && T::B_N == 256 && T::B_K == 128);
    static_assert(T::NUM_WAVES == 8 && T::BLOCK_SIZE == 512 && T::NUM_STAGES == 2);
    static_assert(T::SCALE_PANEL == 12 && T::SFA_BYTES == 2304 && T::SFB_BYTES == 24);
    static_assert(T::MATRIX_LDS_BYTES == 118272 && T::LDS_BYTES == 120600);
    constexpr int FixedK = T::FIXED_K;
    cpu_check::CoordinateGlobal g_a(T::B_M * FixedK), g_b(T::B_N * FixedK);
    cpu_check::CoordinateLDS s_a(T::NUM_STAGES * T::A_STAGE), s_b(T::NUM_STAGES * T::B_STAGE);
    std::uint64_t matrix_producer_bytes = 0, matrix_consumer_bytes = 0;
    std::uint64_t bounded_a_values = 0, bounded_b_values = 0;
    static_for<T::K_TILES>([&](auto tile) {
        constexpr int group = decltype(tile)::value, slot = group % T::NUM_STAGES;
        s_a.reset_slot(slot * T::A_STAGE, T::A_STAGE);
        s_b.reset_slot(slot * T::B_STAGE, T::B_STAGE);
        for (int cpu_thread = 0; cpu_thread < T::BLOCK_SIZE; ++cpu_thread) {
            const int lane_id = cpu_thread % 64;
            const int wave_id_m = (cpu_thread / 64) % T::T_M, wave_id_n = (cpu_thread / 64) / T::T_M;
            struct { int stride_a, stride_b; } kargs{FixedK, FixedK};
#include "host_matrix_layouts.inc"
#include "host_matrix_prefetch.inc"
            s_a.lane = s_b.lane = lane_id;
            // The pair specialization uses runtime groups for its ten refills.
            if constexpr (LoopUnroll == 2 && group >= 2)
                issue_matrix_prefetch(number<slot>{}, int(group));
            else
                issue_matrix_prefetch(number<slot>{}, tile);
        }
        assert(std::count(s_a.writes.begin() + slot * T::A_STAGE,
                          s_a.writes.begin() + (slot + 1) * T::A_STAGE, 1) == T::B_M * T::B_K);
        assert(std::count(s_b.writes.begin() + slot * T::B_STAGE,
                          s_b.writes.begin() + (slot + 1) * T::B_STAGE, 1) == T::B_N * T::B_K);
        matrix_producer_bytes += (T::B_M + T::B_N) * T::B_K;
        for (int cpu_thread = 0; cpu_thread < T::BLOCK_SIZE; ++cpu_thread) {
            const int lane_id = cpu_thread % 64;
            const int wave_id_m = (cpu_thread / 64) % T::T_M, wave_id_n = (cpu_thread / 64) / T::T_M;
            struct { int stride_a, stride_b; } kargs{FixedK, FixedK};
#include "host_matrix_layouts.inc"
            std::array<std::array<int, 16>, T::E_M * 2> a_chunks;
            std::array<std::array<int, 16>, T::E_N * 2> b_chunks;
#include "host_matrix_read.inc"
            static_for<T::E_M>([&](auto mi) { load_a(mi, number<slot>{}); });
            static_for<T::E_N>([&](auto ni) { load_b(ni, number<slot>{}); });
            for (int mi = 0; mi < T::E_M; ++mi) for (int chunk = 0; chunk < 2; ++chunk)
                for (int j = 0; j < 16; ++j) {
                    const int local_row = mi * T::T_M * 16 + wave_id_m * 16 + lane_id % 16;
                    const int k = group * 128 + (lane_id / 16) * 16 + chunk * 64 + j;
                    const int source = a_chunks[mi * 2 + chunk][j];
                    assert(source == local_row * FixedK + k);
                    for (int valid_rows : valid_m_rows) for (int row : {0, 192}) {
                        const int m = row + valid_rows, absolute = row * FixedK + source;
                        assert((absolute < m * FixedK) == (local_row < valid_rows));
                        assert((source < (m - row) * FixedK) == (local_row < valid_rows));
                        ++bounded_a_values;
                    }
                    ++matrix_consumer_bytes;
                }
            for (int ni = 0; ni < T::E_N; ++ni) for (int chunk = 0; chunk < 2; ++chunk)
                for (int j = 0; j < 16; ++j) {
                    const int n_group = ni * T::T_N + wave_id_n;
                    const int expected = n_group * 16 * FixedK + group * 128 * 16 +
                                         chunk * 64 * 16 + lane_id * 16 + j;
                    const int source = b_chunks[ni * 2 + chunk][j];
                    assert(source == expected);
                    const int local_n = n_group * 16 + lane_id % 16;
                    const int scale_half = ni / (T::GROUP_N / (T::T_N * T::W_N));
                    assert(local_n / T::GROUP_N == scale_half);
                    for (int n : n_values) for (int col : {0, n - T::B_N}) {
                        const int absolute = col * FixedK + source, global_n = col + local_n;
                        assert(absolute >= 0 && absolute < n * FixedK);
                        assert(global_n >= col && global_n < col + T::B_N);
                        assert(global_n / T::GROUP_N == col / T::GROUP_N + scale_half);
                        ++bounded_b_values;
                    }
                    ++matrix_consumer_bytes;
                }
        }
        for (int i = slot * T::A_STAGE; i < (slot + 1) * T::A_STAGE; ++i)
            assert(s_a.reads[i] == 2 * s_a.writes[i]);
        for (int i = slot * T::B_STAGE; i < (slot + 1) * T::B_STAGE; ++i)
            assert(s_b.reads[i] == 4 * s_b.writes[i]);
    });
    for (int count : g_a.reads) assert(count == 1);
    for (int count : g_b.reads) assert(count == 1);

    // Claim every byte, including matrix padding, in both physical LDS slots
    // and the complete compact scale panels.
    std::vector<int> lds_owner(T::LDS_BYTES, 0);
    auto claim = [&](int begin, int size) {
        for (int i = begin; i < begin + size; ++i) {
            assert(i >= 0 && i < T::LDS_BYTES);
            assert(++lds_owner[i] == 1);
        }
    };
    for (int stage = 0; stage < T::NUM_STAGES; ++stage) {
        claim(stage * T::A_STAGE, T::A_STAGE);
        claim(T::NUM_STAGES * T::A_STAGE + stage * T::B_STAGE, T::B_STAGE);
    }
    claim(T::MATRIX_LDS_BYTES, T::SFA_BYTES);
    claim(T::MATRIX_LDS_BYTES + T::SFA_BYTES, T::SFB_BYTES);
    for (int count : lds_owner) assert(count == 1);

    auto mma = make_tiled_mma<fp8_t, fp8_t, fp32_t>(
        seq<T::E_M, T::E_N, 1>{}, seq<T::T_M, T::T_N, 1>{},
        seq<16, 16, 128>{}, mfma_adaptor_swap_ab{});
    std::uint64_t output_coordinates = 0, bounded_vectors = 0;
    for (int n : n_values) {
        struct { int stride_c; } kargs{n};
        std::vector<int> covered(T::B_M * T::B_N, 0);
        for (int cpu_thread = 0; cpu_thread < T::BLOCK_SIZE; ++cpu_thread) {
            const int lane_id = cpu_thread % 64;
            const int wave_id_m = (cpu_thread / 64) % T::T_M, wave_id_n = (cpu_thread / 64) / T::T_M;
#include "host_output_layout.inc"
            assert(offsets.size() == T::E_M * T::E_N);
            for (int i = 0; i < offsets.size(); ++i) {
                for (int j = 0; j < T::VEC_C; ++j) {
                    const int off = offsets[i] + j, local_row = off / n, local_col = off % n;
                    assert(local_row >= 0 && local_row < T::B_M && local_col >= 0 && local_col < T::B_N);
                    ++covered[local_row * T::B_N + local_col];
                    ++output_coordinates;
                }
                for (int valid_rows : valid_m_rows) for (int row : {0, 192})
                    for (int col : {0, n - T::B_N}) {
                        const int m = row + valid_rows, local_row = offsets[i] / n;
                        const int absolute = row * n + col + offsets[i];
                        const int byte_bound = ((m - row) * n - col) * 2;
                        const bool first_in_bounds = offsets[i] * 2 < byte_bound;
                        const bool last_in_bounds = (offsets[i] + T::VEC_C) * 2 <= byte_bound;
                        assert((absolute < m * n) == (local_row < valid_rows));
                        assert(first_in_bounds == (local_row < valid_rows));
                        assert(first_in_bounds == last_in_bounds);
                        ++bounded_vectors;
                    }
            }
        }
        for (int count : covered) assert(count == 1);
    }
    std::cout << "K=" << FixedK << " LoopUnroll=" << LoopUnroll
              << " matrix_producer_bytes=" << matrix_producer_bytes
              << " matrix_consumer_bytes=" << matrix_consumer_bytes
              << " bounded_A_values=" << bounded_a_values << " bounded_B_values=" << bounded_b_values
              << " output_coordinates=" << output_coordinates << " bounded_output_vectors=" << bounded_vectors
              << " LDS_bytes=" << T::LDS_BYTES << '\n';
}

#include "scale_schedule_check.inc"

int main() {
    check_matrices_and_output<12>();
    check_matrices_and_output<2>();
    check_scales<12>();
    check_scales<2>();
    const auto full = check_schedule<12>();
    const auto pair = check_schedule<2>();
    assert(full == pair);
    std::cout << "identical_schedule_events=" << full.size() << '\n';
    std::cout << "PASS: CPU-only K1536 production layouts/lambdas, exact scales, M64/M128/M192 tails, "
                 "N through 16384, first/last N tiles, and both identical fixed-K schedules.\n";
}
