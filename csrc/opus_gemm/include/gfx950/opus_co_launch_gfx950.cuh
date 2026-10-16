// SPDX-License-Identifier: Apache-2.0
#pragma once
#if !defined(__HIP_DEVICE_COMPILE__) && !defined(__HIPCC_RTC__)
#include <hip/hip_runtime.h>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>
#include <cstdint>
#include <cstddef>
#include <optional>
#include "aiter_hip_common.h"
#include "aiter_tensor.h"
#include "aiter_stream.h"
#ifndef OPUS_GEN_CO_DIR
#define OPUS_GEN_CO_DIR ""
#endif
namespace opus_gfx950_co {
struct CoModule {
    // Keep the CO bytes alive until the registered kernel is destroyed.
    std::vector<char> image;
    AiterAsmKernelFast kernel;

    CoModule(const std::string& image_name, const std::string& symbol)
        : image(read_co(image_name)),
          kernel(symbol.c_str(), image.data(), AiterAsmKernelFast::SkipGfx1250Gate{}) {}

private:
    static std::vector<char> read_co(const std::string& image_name) {
        const char* env = std::getenv("OPUS_GEN_CO_DIR");
        const std::string root = env && *env ? env : OPUS_GEN_CO_DIR;
        AITER_CHECK(!root.empty(), "gfx950 CO directory is empty");
        const auto path = root + "/gfx950/" + image_name + ".co";
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        AITER_CHECK(file.is_open(), "gfx950 cannot open CO: ", path);
        const std::streamsize size = file.tellg();
        AITER_CHECK(size > 4, "gfx950 invalid CO size: ", path);
        file.seekg(0, std::ios::beg);
        std::vector<char> bytes(static_cast<size_t>(size));
        AITER_CHECK(file.read(bytes.data(), size), "gfx950 short CO read: ", path);
        AITER_CHECK(bytes[0] == 0x7f && bytes[1] == 'E' && bytes[2] == 'L' && bytes[3] == 'F',
                    "gfx950 CO is not a bare ELF code object: ", path);
        return bytes;
    }
};
inline AiterAsmKernelFast& co_kernel(const std::string& image, const std::string& symbol) {
    static SynchronizedCache<std::string, CoModule> cache;
    const std::string key = image + ":" + symbol;
    return cache.get_or_create(key, [&]() { return CoModule(image, symbol); }).kernel;
}
inline AiterAsmKernelFast& co_kernel(const std::string& symbol) { return co_kernel(symbol, symbol); }
} // namespace opus_gfx950_co
template<class Kargs>
inline void opus_co_launch_gfx950(AiterAsmKernelFast& kernel, dim3 grid, dim3 block,
                                  Kargs& args, hipStream_t stream, unsigned dynamic_lds = 0) {
    static_assert(sizeof(args) == 96);
    size_t arg_size = sizeof(args);
    kernel.launch_kernel({&args, &arg_size,
                          int(grid.x), int(grid.y), int(grid.z),
                          int(block.x), int(block.y), int(block.z),
                          stream, 1, 1, 1, dynamic_lds});
}

namespace opus_gfx950_co {
struct LaunchContract {
    int64_t m, n, k;
    unsigned gx, gy, gz, bx, by, bz, dynamic_lds;
};
struct Kargs {
    const void* ptr_a; const void* ptr_b; void* ptr_c;
    int m, n, k, batch, stride_a, stride_b, stride_c;
    int stride_a_batch, stride_b_batch, stride_c_batch;
    const void* ptr_sfa; const void* ptr_sfb;
    int stride_sfa, stride_sfb, stride_sfa_batch, stride_sfb_batch;
};
static_assert(sizeof(Kargs) == 96);
static_assert(offsetof(Kargs, ptr_sfa) == 64 && offsetof(Kargs, stride_sfa) == 80);

template<size_t Count>
inline void launch(AiterAsmKernelFast& kernel,
                   const LaunchContract (&contracts)[Count], const unsigned (&alignment)[5],
                   aiter_tensor_t& x, aiter_tensor_t& w,
                   aiter_tensor_t& sfa, aiter_tensor_t& sfb, aiter_tensor_t& out,
                   const std::optional<aiter_tensor_t>& workspace) {
    constexpr const char* entry = "opus_gemm_gfx950_pin_agpr";
    AITER_CHECK((x.dim() == 2 || x.dim() == 3) && w.dim() == x.dim() && out.dim() == x.dim(),
                entry, ": XQ/WQ/Y must have matching rank 2 or 3");
    AITER_CHECK(x.dim() == 2 || (x.size(0) == 1 && w.size(0) == 1 && out.size(0) == 1),
                entry, ": only batch=1 is supported");
    const int64_t m = x.size(-2), n = w.size(-2), k = x.size(-1);
    const LaunchContract* contract = nullptr;
    for (const auto& candidate : contracts)
        if (candidate.m == m && candidate.n == n && candidate.k == k) { contract = &candidate; break; }
    AITER_CHECK(contract, entry, ": shape is outside the saved launch contracts");
    AITER_CHECK(w.size(-1) == k && out.size(-2) == m && out.size(-1) == n,
                entry, ": XQ/WQ/Y shapes do not match");
    AITER_CHECK(x.is_contiguous() && w.is_contiguous() && out.is_contiguous(),
                entry, ": XQ/WQ/Y must be contiguous");
    AITER_CHECK(x.dtype() == AITER_DTYPE_fp8 && w.dtype() == AITER_DTYPE_fp8 &&
                out.dtype() == AITER_DTYPE_bf16, entry, ": requires FP8 inputs and BF16 output");
    const auto e8m0 = [](const aiter_tensor_t& t) {
        return t.dtype() == AITER_DTYPE_fp8_e8m0 || t.dtype() == AITER_DTYPE_u8;
    };
    AITER_CHECK(e8m0(sfa) && e8m0(sfb) && sfa.dim() == 2 && sfb.dim() == 2 &&
                sfa.size(0) == m && sfa.size(1) == k / 128 &&
                sfa.stride(0) == 1 && sfa.stride(1) == m &&
                sfb.size(0) == n / 128 && sfb.size(1) == k / 128 && sfb.is_contiguous(),
                entry, ": E8M0 scale layout differs");
    AITER_CHECK(!workspace.has_value(), entry, ": direct CO does not use workspace");
    const aiter_tensor_t* tensors[] = {&x, &w, &out, &sfa, &sfb};
    const uintptr_t output = reinterpret_cast<uintptr_t>(out.data_ptr());
    const uint64_t output_bytes = uint64_t(m) * n * 2;
    for (size_t i = 0; i < 5; ++i) {
        const uintptr_t begin = reinterpret_cast<uintptr_t>(tensors[i]->data_ptr());
        AITER_CHECK(!alignment[i] || begin % alignment[i] == 0, entry, ": pointer alignment differs");
        if (i == 2) continue;
        const uint64_t bytes = tensors[i]->numel() * tensors[i]->element_size();
        const bool overlap = output >= begin ? output - begin < bytes : begin - output < output_bytes;
        AITER_CHECK(!overlap, entry, ": Y must not overlap input storage");
    }
    Kargs args{};
    args.ptr_a = x.data_ptr(); args.ptr_b = w.data_ptr(); args.ptr_c = out.data_ptr();
    args.m = int(m); args.n = int(n); args.k = int(k); args.batch = 1;
    args.stride_a = int(k); args.stride_b = int(k); args.stride_c = int(n);
    args.stride_a_batch = int(m * k); args.stride_b_batch = int(n * k); args.stride_c_batch = int(m * n);
    args.ptr_sfa = sfa.data_ptr(); args.ptr_sfb = sfb.data_ptr();
    args.stride_sfa = int(m); args.stride_sfb = int(k / 128);
    args.stride_sfa_batch = int(m * (k / 128)); args.stride_sfb_batch = int((n / 128) * (k / 128));
    opus_co_launch_gfx950(kernel, dim3(contract->gx, contract->gy, contract->gz),
                         dim3(contract->bx, contract->by, contract->bz), args,
                         aiter::getCurrentHIPStream(), contract->dynamic_lds);
}
} // namespace opus_gfx950_co
#endif
