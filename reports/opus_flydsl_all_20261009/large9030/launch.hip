#include <hip/hip_runtime.h>
#include <cstdint>
#define __HIPCC_RTC__ 1
#include "contract.h"
#if defined(PRIVATE9030_DIRECTB)
#include "directb/pipeline.cuh"
#elif defined(PRIVATE9030_PANEL16)
#include "panel16/pipeline.cuh"
#else
#include "opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_8wave_192x256_large_output_gfx950.cuh"
#endif

extern "C" __attribute__((visibility("default")))
int launch(int kid, const void* a, const void* b, const void* sfa, const void* sfb,
           void* c, int m, int n, int k, void* stream) {
#if defined(PRIVATE9030_DIRECTB) || defined(PRIVATE9030_PANEL16)
    constexpr bool fixed_only = true;
#else
    constexpr bool fixed_only = false;
#endif
    if (kid != 9030 || !private9030_shape_valid(m, n, k, fixed_only) ||
        !private9030_pointer_valid(a, b, sfa, sfb, c))
        return int(hipErrorInvalidValue);
    opus_gemm_mxscale_bpreshuffle_kargs_gfx950 args{};
    args.ptr_a = a; args.ptr_b = b; args.ptr_sfa = sfa; args.ptr_sfb = sfb; args.ptr_c = c;
    args.m = m; args.n = n; args.k = k; args.batch = 1;
    args.stride_a = k; args.stride_b = k; args.stride_c = n;
    args.stride_sfa = m; args.stride_sfb = k / 128;
    args.stride_a_batch = m * k; args.stride_b_batch = n * k;
    args.stride_c_batch = 0; // Batch=1; never narrow the large total C element count.
    args.stride_sfa_batch = m * (k / 128); args.stride_sfb_batch = (n / 128) * (k / 128);
    const dim3 grid(n / 256, 1 + (m - 1) / 192);
    const auto hip_stream = reinterpret_cast<hipStream_t>(stream);
#if defined(PRIVATE9030_DIRECTB)
    using T = opus_private_9030_directb_a3_chunk96_traits;
    opus_private_9030_directb_a3_chunk96_kernel<T><<<grid, T::BLOCK_SIZE, 0, hip_stream>>>(args);
#elif defined(PRIVATE9030_PANEL16)
    using T = opus_private_9030_fixed1536_panel16_traits;
    opus_private_9030_fixed1536_panel16_kernel<T><<<grid, T::BLOCK_SIZE, 0, hip_stream>>>(args);
#else
    using T = opus_gemm_mxscale_bpreshuffle_8wave_192x256_large_output_traits_gfx950;
    gemm_a8w8_mxfp8_scale_8wave_192x256_large_output_kernel<T><<<grid, T::BLOCK_SIZE, 0, hip_stream>>>(args);
#endif
    return int(hipGetLastError());
}
