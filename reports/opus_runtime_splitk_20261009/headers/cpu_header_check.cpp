#include "gfx950/opus_gemm_mxscale_bpreshuffle_runtime_splitk_helpers_gfx950.cuh"
#include "gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_runtime_splitk_gfx950.cuh"
#include "actual_b_layout_helpers.h"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

using opus_gemm_mxscale_bpreshuffle_runtime_detail::balanced_partition;
using Args = opus_gemm_mxscale_bpreshuffle_runtime_kargs_gfx950;
using LegacyArgs = opus_gemm_mxscale_bpreshuffle_kargs_gfx950;

void partition_check() {
    int cases = 0, nested_cases = 0;
    for (int total = 0; total <= 128; ++total) {
        for (int parts = 1; parts <= 16; ++parts) {
            std::vector<int> coverage(total), nested(total);
            int end = 0, smallest = total, largest = 0;
            for (int i = 0; i < parts; ++i) {
                const auto p = balanced_partition(total, parts, i);
                assert(p.begin == end && p.count >= 0 && p.begin + p.count <= total);
                end = p.begin + p.count;
                smallest = std::min(smallest, p.count);
                largest = std::max(largest, p.count);
                for (int k = p.begin; k < end; ++k) ++coverage[k];
                int local_end = 0;
                for (int wk = 0; wk < 2; ++wk) {
                    const auto q = balanced_partition(p.count, 2, wk);
                    assert(q.begin == local_end && q.count >= 0);
                    local_end = q.begin + q.count;
                    for (int k = q.begin; k < local_end; ++k) ++nested[p.begin + k];
                }
                assert(local_end == p.count);
                ++nested_cases;
            }
            assert(end == total && largest - smallest <= 1);
            for (int k = 0; k < total; ++k) assert(coverage[k] == 1 && nested[k] == 1);
            ++cases;
        }
    }
    Args args{};
    args.m = 16; args.k = 128; args.split_k = 3;
    assert(static_cast<LegacyArgs*>(&args)->m == 16);
    assert(static_cast<LegacyArgs*>(&args)->k == 128);
    assert(reinterpret_cast<const char*>(&args) == reinterpret_cast<const char*>(static_cast<const LegacyArgs*>(&args)));
    assert(reinterpret_cast<const char*>(&args.split_k) >= reinterpret_cast<const char*>(&args) + sizeof(LegacyArgs));
    std::cout << "partition_cases=" << cases << " nested_local_cases=" << nested_cases << '\n';
}

template<class T>
void fine_layout_check() {
    // Compile the actual production B layout expressions with host attributes.
    // Every byte copied to LDS is identified by its packed global B byte.
    std::vector<int> b_source(T::B_STAGE, -1), b_copies(T::B_STAGE);
    const int stride_b = 16384;
    for (int wave = 0; wave < T::NUM_WAVES; ++wave) {
        const int wm = wave % T::T_M, wn = wave / T::T_M;
        for (int lane = 0; lane < 64; ++lane) {
            const auto go = opus::layout_to_offsets<16>(checked_layout::make_layout_gb_scale<T>(lane, wm, wn, stride_b));
            const auto so = opus::layout_to_offsets<16>(checked_layout::make_layout_sb_scale<T>(wm, wn));
            for (int copy = 0; copy < T::B_N * 128 / (T::BLOCK_SIZE * 16); ++copy) {
                for (int byte = 0; byte < 16; ++byte) {
                    const int address = so[copy] + lane * 16 + byte;
                    assert(address >= 0 && address < T::B_STAGE);
                    b_source[address] = go[copy] + byte;
                    assert(++b_copies[address] == 1);
                }
            }
        }
    }
    for (int wave = 0; wave < T::NUM_WAVES; ++wave) {
        const int wn = wave / T::T_M;
        for (int lane = 0; lane < 64; ++lane) {
            const auto ro = opus::layout_to_offsets<16>(checked_layout::make_layout_rb_scale<T>(lane, wn));
            for (int fragment = 0; fragment < T::E_N; ++fragment) {
                for (int half = 0; half < 2; ++half) {
                    const int address = ro[fragment * 2 + half];
                    const int n = (fragment * T::T_N + wn) * 16;
                    const int expected = n * stride_b + lane * 16 + half * 1024;
                    for (int byte = 0; byte < 16; ++byte) {
                        assert(address + byte < T::B_STAGE && b_copies[address + byte] == 1);
                        assert(b_source[address + byte] == expected + byte);
                    }
                }
            }
        }
    }
    // Fine-M copies may use fewer waves on their last pass. Audit the complete
    // A byte map and every MFMA operand read, including padding between groups.
    std::vector<int> a_source(T::A_STAGE, -1), a_copies(T::A_STAGE);
    for (int group = 0; group < T::B_M / 8; ++group) {
        const int wm = (group % T::NUM_WAVES) % T::T_M;
        for (int lane = 0; lane < 64; ++lane) {
            const int r = (group / T::T_M) * 8 * T::T_M + (lane / 8) * T::T_M + wm;
            for (int byte = 0; byte < 16; ++byte) {
                const int address = group * (1024 + T::smem_padding) + lane * 16 + byte;
                assert(r < T::B_M && address < T::A_STAGE);
                a_source[address] = r * 128 + (lane % 8) * 16 + byte;
                assert(++a_copies[address] == 1);
            }
        }
    }
    for (int wm = 0; wm < T::T_M; ++wm) {
        for (int lane = 0; lane < 64; ++lane) {
            for (int fragment = 0; fragment < T::E_M; ++fragment) {
                const int r = (fragment * T::T_M + wm) * 16 + lane % 16;
                const int offset = ((r / (8 * T::T_M)) * T::T_M + r % T::T_M) * (1024 + T::smem_padding) +
                    ((r / T::T_M) % 8) * 128 + (lane / 16) * 16;
                for (int half = 0; half < 2; ++half) {
                    for (int byte = 0; byte < 16; ++byte) {
                        const int address = offset + half * 64 + byte;
                        assert(a_copies[address] == 1);
                        assert(a_source[address] == r * 128 + (lane / 16) * 16 + half * 64 + byte);
                    }
                }
            }
        }
    }
    int lds_cases = 0;
    for (int tiles = 0; tiles <= 128; ++tiles) {
        for (int splits = 1; splits <= 16; ++splits) {
            const int capacity = T::lds_bytes(tiles * 128, splits);
            assert(capacity <= T::MAX_LDS_BYTES);
            for (int split = 0; split < splits; ++split) {
                const auto p = balanced_partition(tiles, splits, split);
                const int stages = std::min(p.count, T::NUM_STAGES);
                const int a_end = stages * T::A_STAGE;
                const int b_end = a_end + stages * T::B_STAGE;
                const int sfa_end = b_end + T::B_M * p.count;
                const int sfb_end = sfa_end + T::B_GROUPS * p.count;
                assert(sfb_end <= capacity);
                assert(T::B_GROUPS * p.count <= T::BLOCK_SIZE);
                assert(T::B_M * p.count <= T::MAX_SFA_BYTES);
            }
            ++lds_cases;
        }
    }
    assert(T::lds_bytes(16384, 1) == T::MAX_LDS_BYTES);
    std::cout << "fine=" << T::B_M << 'x' << T::B_N << " lds_cases=" << lds_cases << " max_lds=" << T::MAX_LDS_BYTES << '\n';
}

template<class T>
void register_layout_check() {
    static_assert(T::REDUCE_VEC == 16 && T::REDUCE_BLOCK == 128);
    // Local-wave partial storage covers each fragment once. Wave zero consumes
    // the same address after the barrier, even when a K wave has no input tile.
    if constexpr (T::WAVE_K > 1) {
        constexpr int elems = T::B_M * T::B_N;
        std::vector<int> writes((T::WAVE_K - 1) * elems);
        for (int wk = 1; wk < T::WAVE_K; ++wk) {
            for (int lane = 0; lane < 64; ++lane) {
                for (int fragment = 0; fragment < T::E_M * T::E_N; ++fragment) {
                    for (int elem = 0; elem < 4; ++elem) {
                        const int offset = (wk - 1) * elems + lane * 4 + fragment * 256 + elem;
                        assert(offset < int(writes.size()));
                        assert(++writes[offset] == 1);
                    }
                }
            }
        }
        for (int n : writes) assert(n == 1);
    }
    std::cout << "register=" << T::B_M << 'x' << T::B_N << " local_wave_k=" << T::WAVE_K << '\n';
}

int main() {
    partition_check();
    fine_layout_check<opus_gemm_mxscale_bpreshuffle_narrow_fine_runtime_traits<48,64,1,4>>();
    fine_layout_check<opus_gemm_mxscale_bpreshuffle_narrow_fine_runtime_traits<64,128,2,2>>();
    fine_layout_check<opus_gemm_mxscale_bpreshuffle_narrow_fine_runtime_traits<96,128,2,2>>();
    register_layout_check<opus_gemm_mxscale_bpreshuffle_register_runtime_traits<16,16,1>>();
    register_layout_check<opus_gemm_mxscale_bpreshuffle_register_runtime_traits<16,16,2>>();
    register_layout_check<opus_gemm_mxscale_bpreshuffle_register_runtime_traits<16,32,1>>();
    register_layout_check<opus_gemm_mxscale_bpreshuffle_register_runtime_traits<16,32,2>>();
    register_layout_check<opus_gemm_mxscale_bpreshuffle_register_runtime_traits<32,32,1>>();
    register_layout_check<opus_gemm_mxscale_bpreshuffle_register_runtime_traits<32,64,1>>();
}
