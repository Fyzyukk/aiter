// SPDX-License-Identifier: Apache-2.0
#pragma once

// Instantiate the existing four-wave schedule with in-kernel M padding.
// This is one GEMM launch, with no separate tail kernel or padded tensor.
#include "opus_gemm_traits_a8w8_mxscale_bpreshuffle_padded_m_gfx950.cuh"
#include "opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_4wave_gfx950.cuh"
