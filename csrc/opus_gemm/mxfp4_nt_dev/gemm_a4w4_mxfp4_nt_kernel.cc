#include <opus/hip_minimal.hpp>

#include "opus_gemm_pipeline_a4w4_mxfp4_nt_gfx950.cuh"

template __global__ void gemm_a4w4_mxfp4_nt_kernel<opus_mxfp4_nt_traits<256, 256, 0>>(opus_gemm_mxfp4_scale_kargs_gfx950);
template __global__ void gemm_a4w4_mxfp4_nt_kernel<opus_mxfp4_nt_traits<256, 128, 0>>(opus_gemm_mxfp4_scale_kargs_gfx950);
template __global__ void gemm_a4w4_mxfp4_nt_kernel<opus_mxfp4_nt_traits<128, 256, 0>>(opus_gemm_mxfp4_scale_kargs_gfx950);
template __global__ void gemm_a4w4_mxfp4_nt_kernel<opus_mxfp4_nt_traits<128, 128, 0>>(opus_gemm_mxfp4_scale_kargs_gfx950);
template __global__ void gemm_a4w4_mxfp4_nt_kernel<opus_mxfp4_nt_traits<192, 256, 0>>(opus_gemm_mxfp4_scale_kargs_gfx950);
template __global__ void gemm_a4w4_mxfp4_nt_kernel<opus_mxfp4_nt_traits<256, 192, 0>>(opus_gemm_mxfp4_scale_kargs_gfx950);
template __global__ void gemm_a4w4_mxfp4_nt_kernel<opus_mxfp4_nt_traits<256, 256, 1>>(opus_gemm_mxfp4_scale_kargs_gfx950);
template __global__ void gemm_a4w4_mxfp4_nt_kernel<opus_mxfp4_nt_traits<256, 128, 1>>(opus_gemm_mxfp4_scale_kargs_gfx950);
template __global__ void gemm_a4w4_mxfp4_nt_kernel<opus_mxfp4_nt_traits<128, 256, 1>>(opus_gemm_mxfp4_scale_kargs_gfx950);
template __global__ void gemm_a4w4_mxfp4_nt_kernel<opus_mxfp4_nt_traits<128, 128, 1>>(opus_gemm_mxfp4_scale_kargs_gfx950);
template __global__ void gemm_a4w4_mxfp4_nt_kernel<opus_mxfp4_nt_traits<192, 256, 1>>(opus_gemm_mxfp4_scale_kargs_gfx950);
template __global__ void gemm_a4w4_mxfp4_nt_kernel<opus_mxfp4_nt_traits<256, 192, 1>>(opus_gemm_mxfp4_scale_kargs_gfx950);
template __global__ void gemm_a4w4_mxfp4_nt_kernel<opus_mxfp4_nt_traits<256, 128, 2>>(opus_gemm_mxfp4_scale_kargs_gfx950);
template __global__ void gemm_a4w4_mxfp4_nt_kernel<opus_mxfp4_nt_traits<128, 256, 2>>(opus_gemm_mxfp4_scale_kargs_gfx950);
template __global__ void gemm_a4w4_mxfp4_nt_kernel<opus_mxfp4_nt_traits<128, 128, 2>>(opus_gemm_mxfp4_scale_kargs_gfx950);
template __global__ void gemm_a4w4_mxfp4_nt_kernel<opus_mxfp4_nt_traits<192, 256, 2>>(opus_gemm_mxfp4_scale_kargs_gfx950);
template __global__ void gemm_a4w4_mxfp4_nt_kernel<opus_mxfp4_nt_traits<256, 192, 2>>(opus_gemm_mxfp4_scale_kargs_gfx950);
template __global__ void gemm_a4w4_mxfp4_nt_kernel<opus_mxfp4_nt_traits<256, 256, 0, false>>(opus_gemm_mxfp4_scale_kargs_gfx950);
template __global__ void gemm_a4w4_mxfp4_nt_sk_reduce<opus_mxfp4_nt_traits<256, 128, 2>>(opus_gemm_mxfp4_scale_kargs_gfx950);
template __global__ void gemm_a4w4_mxfp4_nt_sk_reduce<opus_mxfp4_nt_traits<128, 256, 2>>(opus_gemm_mxfp4_scale_kargs_gfx950);
template __global__ void gemm_a4w4_mxfp4_nt_sk_reduce<opus_mxfp4_nt_traits<128, 128, 2>>(opus_gemm_mxfp4_scale_kargs_gfx950);
template __global__ void gemm_a4w4_mxfp4_nt_sk_reduce<opus_mxfp4_nt_traits<192, 256, 2>>(opus_gemm_mxfp4_scale_kargs_gfx950);
template __global__ void gemm_a4w4_mxfp4_nt_sk_reduce<opus_mxfp4_nt_traits<256, 192, 2>>(opus_gemm_mxfp4_scale_kargs_gfx950);
