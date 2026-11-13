#include <opus/hip_minimal.hpp>

#include "opus_gemm_pipeline_a4w4_mxfp4_nt_gfx950.cuh"

template __global__ void gemm_a4w4_mxfp4_nt_kernel<opus_mxfp4_nt_traits<256, 256>>(opus_gemm_mxfp4_scale_kargs_gfx950);
template __global__ void gemm_a4w4_mxfp4_nt_kernel<opus_mxfp4_nt_traits<256, 128>>(opus_gemm_mxfp4_scale_kargs_gfx950);
template __global__ void gemm_a4w4_mxfp4_nt_kernel<opus_mxfp4_nt_traits<128, 256>>(opus_gemm_mxfp4_scale_kargs_gfx950);
template __global__ void gemm_a4w4_mxfp4_nt_kernel<opus_mxfp4_nt_traits<128, 128>>(opus_gemm_mxfp4_scale_kargs_gfx950);
