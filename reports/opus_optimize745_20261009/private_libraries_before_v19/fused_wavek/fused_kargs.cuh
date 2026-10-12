// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "gfx950/opus_gemm_mxscale_bpreshuffle_runtime_kargs_gfx950.cuh"
struct opus745_fused_wavek_kargs : opus_gemm_mxscale_bpreshuffle_runtime_kargs_gfx950 {
    float* ptr_workspace;
    int* ptr_counters;
};
