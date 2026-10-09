// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "opus_gemm_traits_a8w8_mxscale_bpreshuffle_small_gfx950.cuh"

// K128 tiles are balanced between workgroups. Every partial, including empty
// partitions, is overwritten before a separate FP32 reduction converts to BF16.
template<int BlockM, int WaveM, int WaveN, int Stages, int Cluster, int SplitK,
         int ReduceVec=4, int ReduceBlock=128, int StoreCache=0, int FixedK=0>
struct opus_gemm_mxscale_bpreshuffle_fine_traits_gfx950
    : opus_gemm_small_lds_traits_gfx950<
          BlockM, 128, WaveM, WaveN, Stages, Cluster, 2,
          false, false, true, true, true,
          SplitK, ReduceVec, ReduceBlock, StoreCache, FixedK, true> {};
