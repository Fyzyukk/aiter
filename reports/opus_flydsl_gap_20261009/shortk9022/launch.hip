#include <hip/hip_runtime.h>
#include <cstdint>
#define __HIPCC_RTC__ 1
#include "contract.h"
#if defined(SHORTK9022_CANDIDATE)
#include "candidate/pipeline.cuh"
#else
#include "opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_4wave_160x128_gfx950.cuh"
#endif

extern "C" __attribute__((visibility("default")))
int launch(int kid, const void* a, const void* b, const void* sfa, const void* sfb,
           void* c, int m, int n, int k, void* stream) {
#if defined(SHORTK9022_CANDIDATE)
    constexpr bool fixed_only = true;
#else
    constexpr bool fixed_only = false;
#endif
    if (kid != 9022 || !shortk9022_shape_valid(m, n, k, fixed_only) ||
        !shortk9022_pointer_valid(a, b, sfa, sfb, c))
        return int(hipErrorInvalidValue);
    opus_gemm_mxscale_bpreshuffle_kargs_gfx950 args{};
    args.ptr_a = a; args.ptr_b = b; args.ptr_sfa = sfa; args.ptr_sfb = sfb; args.ptr_c = c;
    args.m = m; args.n = n; args.k = k; args.batch = 1;
    args.stride_a = k; args.stride_b = k; args.stride_c = n;
    args.stride_sfa = m; args.stride_sfb = k / 128;
    args.stride_a_batch = m * k; args.stride_b_batch = n * k; args.stride_c_batch = m * n;
    args.stride_sfa_batch = m * (k / 128); args.stride_sfb_batch = (n / 128) * (k / 128);
    const dim3 grid(n / 128, 1 + (m - 1) / 160);
    const auto hip_stream = reinterpret_cast<hipStream_t>(stream);
#if defined(SHORTK9022_CANDIDATE)
    if (k == 384) {
        using T = opus_private_shortk9022_panel8_traits<384>;
        opus_private_shortk9022_panel8_kernel<T><<<grid, T::BLOCK_SIZE, 0, hip_stream>>>(args);
    } else {
        using T = opus_private_shortk9022_panel8_traits<768>;
        opus_private_shortk9022_panel8_kernel<T><<<grid, T::BLOCK_SIZE, 0, hip_stream>>>(args);
    }
#else
    using T = opus_gemm_mxscale_bpreshuffle_4wave_160x128_traits_gfx950;
    gemm_a8w8_mxfp8_scale_4wave_160x128_kernel<T><<<grid, T::BLOCK_SIZE, 0, hip_stream>>>(args);
#endif
    return int(hipGetLastError());
}
