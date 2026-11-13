#include <opus/hip_minimal.hpp>

#include "opus_gemm_pipeline_a4w4_mxfp4_nt_gfx950.cuh"

template __global__ void gemm_a4w4_mxfp4_4wave_kernel<gemm_a4w4_mxfp4_4wave_traits_t<256, 256>>(opus_gemm_mxfp4_scale_kargs);
template __global__ void gemm_a4w4_mxfp4_4wave_kernel<gemm_a4w4_mxfp4_4wave_traits_t<256, 128>>(opus_gemm_mxfp4_scale_kargs);
template __global__ void gemm_a4w4_mxfp4_4wave_kernel<gemm_a4w4_mxfp4_4wave_traits_t<128, 256>>(opus_gemm_mxfp4_scale_kargs);
template __global__ void gemm_a4w4_mxfp4_4wave_kernel<gemm_a4w4_mxfp4_4wave_traits_t<128, 128>>(opus_gemm_mxfp4_scale_kargs);
