// CPU-built, isolated register-scale experiment; no production dispatcher.
#if !defined(__HIP_DEVICE_COMPILE__)
#include <hip/hip_runtime.h>
#endif
#include <opus/hip_minimal.hpp>
#include <cstdint>
#include <climits>
#include "opus_gemm_traits_a8w8_mxscale_bpreshuffle_small_gfx950.cuh"

#include "opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_small_register_gfx950.cuh"
template __global__ void gemm_a8w8_mxfp8_scale_small_register_kernel<opus_gemm_mxscale_bpreshuffle_small_register_16x32_traits_gfx950>(opus_gemm_mxscale_bpreshuffle_kargs_gfx950);
template __global__ void gemm_a8w8_mxfp8_scale_small_register_kernel<opus_gemm_mxscale_bpreshuffle_small_register_16x16_traits_gfx950>(opus_gemm_mxscale_bpreshuffle_kargs_gfx950);
template __global__ void gemm_a8w8_mxfp8_scale_small_register_kernel<opus_gemm_mxscale_bpreshuffle_small_register_32x32_traits_gfx950>(opus_gemm_mxscale_bpreshuffle_kargs_gfx950);
template __global__ void gemm_a8w8_mxfp8_scale_small_register_kernel<opus_gemm_mxscale_bpreshuffle_small_register_prefetch_16x16_traits_gfx950>(opus_gemm_mxscale_bpreshuffle_kargs_gfx950);
template __global__ void gemm_a8w8_mxfp8_scale_small_register_kernel<opus_gemm_mxscale_bpreshuffle_small_register_prefetch_16x32_traits_gfx950>(opus_gemm_mxscale_bpreshuffle_kargs_gfx950);
template __global__ void gemm_a8w8_mxfp8_scale_small_register_kernel<opus_gemm_mxscale_bpreshuffle_small_register_wavek_16x32_traits_gfx950>(opus_gemm_mxscale_bpreshuffle_kargs_gfx950);
template __global__ void gemm_a8w8_mxfp8_scale_small_register_kernel<opus_gemm_mxscale_bpreshuffle_small_register_wavek_32x32_traits_gfx950>(opus_gemm_mxscale_bpreshuffle_kargs_gfx950);
template __global__ void gemm_a8w8_mxfp8_scale_small_register_kernel<opus_gemm_mxscale_bpreshuffle_small_register_wavek_32x64_traits_gfx950>(opus_gemm_mxscale_bpreshuffle_kargs_gfx950);


// All tensors are contiguous native E8M0 inputs with N16/K16 preshuffled B.
// sfa is column-major logical [M,K/128]; sfb is row-major [N/128,K/128].
extern "C" __attribute__((visibility("default")))
int launch(int variant, const void* a, const void* b, const void* sfa, const void* sfb,
           void* c, int m, int n, int k, void* stream) {
    constexpr int64_t limit = INT32_MAX;
    if (m <= 0 || m > 512 || n <= 0 || n % 128 || k <= 0 || k > 16384 || k % 128 ||
        int64_t(m) * k > limit || int64_t(n) * k > limit || int64_t(m) * n * 2 > limit ||
        !a || !b || !sfa || !sfb || !c ||
        reinterpret_cast<uintptr_t>(a) % 16 || reinterpret_cast<uintptr_t>(b) % 16 ||
        reinterpret_cast<uintptr_t>(c) % 16)
        return 1; // hipErrorInvalidValue
    opus_gemm_mxscale_bpreshuffle_kargs_gfx950 args{};
    args.ptr_a=a; args.ptr_b=b; args.ptr_c=c; args.ptr_sfa=sfa; args.ptr_sfb=sfb;
    args.m=m; args.n=n; args.k=k; args.batch=1;
    args.stride_a=k; args.stride_b=k; args.stride_c=n;
    args.stride_a_batch=m*k; args.stride_b_batch=n*k; args.stride_c_batch=m*n;
    args.stride_sfa=m; args.stride_sfb=k/128;
    args.stride_sfa_batch=m*(k/128); args.stride_sfb_batch=(n/128)*(k/128);
    const auto hs = reinterpret_cast<hipStream_t>(stream);
#define RUN(ID, TYPE) case ID: { \
    using T = TYPE; \
    const dim3 grid(n/T::B_N, (m+T::B_M-1)/T::B_M); \
    gemm_a8w8_mxfp8_scale_small_register_kernel<T><<<grid, T::BLOCK_SIZE, 0, hs>>>(args); \
    break; }
    switch (variant) {
    RUN(9040, opus_gemm_mxscale_bpreshuffle_small_register_16x32_traits_gfx950)
    RUN(9041, opus_gemm_mxscale_bpreshuffle_small_register_16x16_traits_gfx950)
    RUN(9042, opus_gemm_mxscale_bpreshuffle_small_register_32x32_traits_gfx950)
    RUN(9050, opus_gemm_mxscale_bpreshuffle_small_register_prefetch_16x16_traits_gfx950)
    RUN(9051, opus_gemm_mxscale_bpreshuffle_small_register_prefetch_16x32_traits_gfx950)
    RUN(9052, opus_gemm_mxscale_bpreshuffle_small_register_wavek_16x32_traits_gfx950)
    RUN(9053, opus_gemm_mxscale_bpreshuffle_small_register_wavek_32x32_traits_gfx950)
    RUN(9054, opus_gemm_mxscale_bpreshuffle_small_register_wavek_32x64_traits_gfx950)
    default: return 1;
    }
#undef RUN
    return static_cast<int>(hipGetLastError());
}
