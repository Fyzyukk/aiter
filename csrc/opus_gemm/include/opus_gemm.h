// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

// Family-specific OPUS GEMM entry points. Uses aiter_tensor_t (POD,
// torch-free) instead of torch::Tensor so this header costs ~200
// preprocessed lines instead of the ~50K that <torch/all.h> +
// <torch/extension.h> drag in. Mirrors the refactor in PR #2932
// (csrc/include/quant.h). The pybind layer registers aiter_tensor_t as a
// pybind11 class for compatibility/A-B calls; the production A16 Python path
// uses the C ABI declaration below with ctypes aiter_tensor_t values.
#include "aiter_tensor.h"
#include <optional>

void opus_gemm_a16w16_launch(aiter_tensor_t& XQ,
                             aiter_tensor_t& WQ,
                             aiter_tensor_t& Y,
                             std::optional<aiter_tensor_t> bias,
                             std::optional<aiter_tensor_t> workspace,
                             int kid,
                             int split_k);

// C ABI used by the production Python launcher after the Torch/pybind boundary
// experiment met its acceptance criteria. The implementation forwards to
// opus_gemm_a16w16_launch(), so it shares the exact-kid dispatch and all
// generated checked validators. Optional tensors are represented by nullptr;
// the caller supplies the live HIP stream, while the wrapper switches to the
// Tensor's HIP device for the call and restores the previous device afterward.
AITER_C_ITFS int opus_gemm_a16w16_launch_cabi(aiter_tensor_t* XQ,
                                              aiter_tensor_t* WQ,
                                              aiter_tensor_t* Y,
                                              aiter_tensor_t* bias,
                                              aiter_tensor_t* workspace,
                                              int64_t kid,
                                              int64_t split_k,
                                              hipStream_t stream);

void opus_gemm_a8w8_launch(aiter_tensor_t& XQ,
                           aiter_tensor_t& WQ,
                           aiter_tensor_t& Y,
                           int kid);

void opus_gemm_a8w8_blockscale_launch(aiter_tensor_t& XQ,
                                      aiter_tensor_t& WQ,
                                      aiter_tensor_t& Y,
                                      aiter_tensor_t& x_scale,
                                      aiter_tensor_t& w_scale,
                                      int kid);

void opus_gemm_a8w8_blockscale_bpreshuffle_launch(
    aiter_tensor_t& XQ,
    aiter_tensor_t& WQ,
    aiter_tensor_t& x_scale,
    aiter_tensor_t& w_scale,
    aiter_tensor_t& Y,
    int kid);
