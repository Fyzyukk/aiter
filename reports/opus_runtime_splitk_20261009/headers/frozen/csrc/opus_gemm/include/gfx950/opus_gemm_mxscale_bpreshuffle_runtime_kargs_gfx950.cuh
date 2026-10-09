// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "opus_gemm_traits_a8w8_mxscale_bpreshuffle_gfx950.cuh"

// The legacy argument type and entry points retain their ABI. Only runtime
// split entry points consume this extension; the launcher resolves split_k.
struct opus_gemm_mxscale_bpreshuffle_runtime_kargs_gfx950
    : opus_gemm_mxscale_bpreshuffle_kargs_gfx950 {
    int split_k;
};
