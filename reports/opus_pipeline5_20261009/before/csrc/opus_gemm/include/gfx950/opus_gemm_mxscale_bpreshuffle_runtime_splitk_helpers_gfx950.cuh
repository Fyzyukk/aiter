// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include <opus/hip_minimal.hpp>
#include <opus/opus.hpp>

namespace opus_gemm_mxscale_bpreshuffle_runtime_detail {

struct partition {
    int begin;
    int count;
};

// The launcher validates parts > 0 and 0 <= index < parts. This also handles
// empty local-wave partitions when global splitting leaves fewer K tiles.
__host__ __device__ constexpr partition balanced_partition(int total, int parts, int index) {
    const int per = total / parts, extra = total % parts;
    return {index * per + (index < extra ? index : extra), per + (index < extra)};
}

} // namespace opus_gemm_mxscale_bpreshuffle_runtime_detail

// One reducer specialization per IO geometry serves every runtime split count.
// Workspace planes contain FP32 local-wave sums; BF16 conversion occurs once.
template<int Vec = 4, int Block = 128>
__global__ __launch_bounds__(Block, 2)
void opus_gemm_mxscale_bpreshuffle_reduce_runtime_kernel(
    const float* __restrict__ workspace, opus::bf16_t* __restrict__ out,
    int elements, int split_k) {
    static_assert(Vec >= 4 && Vec <= 16 && Vec % 4 == 0);
    static_assert(Block > 0);
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
    using namespace opus;
    const int index = (block_id_x() * Block + thread_id_x()) * Vec;
    if (index >= elements) return;
    vector_t<float, Vec> acc{};
    #pragma clang loop unroll(disable)
    for (int split = 0; split < split_k; ++split) {
        auto source = make_gmem(workspace + static_cast<int64_t>(split) * elements,
                                static_cast<unsigned>(elements * sizeof(float)));
        static_for<Vec / 4>([&](auto part) {
            constexpr int offset = decltype(part)::value * 4;
            const auto value = load<4>(source, index + offset);
            static_for<4>([&](auto elem) { acc[offset + decltype(elem)::value] += value[decltype(elem)::value]; });
        });
    }
    auto target = make_gmem(out, static_cast<unsigned>(elements * sizeof(bf16_t)));
    constexpr int store_vec = Vec < 8 ? Vec : 8;
    static_for<Vec / store_vec>([&](auto part) {
        constexpr int offset = decltype(part)::value * store_vec;
        store<store_vec>(target, cast<bf16_t>(slice(acc, number<offset>{}, number<offset+store_vec>{})), index+offset);
    });
#endif
}
