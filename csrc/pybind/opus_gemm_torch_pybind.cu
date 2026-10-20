// SPDX-License-Identifier: MIT
// Tensor metadata conversion for the OPUS BPreshuffle host interface.
#ifndef __HIP_DEVICE_COMPILE__

#include <ATen/core/Tensor.h>
#include <c10/hip/HIPStream.h>
#include <torch/csrc/utils/pybind.h>

#include "aiter_stream.h"
#include "aiter_tensor.h"
#include "opus_gemm.h"

namespace {
aiter_tensor_t opus_tensor_descriptor(const at::Tensor& tensor)
{
    TORCH_CHECK(tensor.dim() <= 8, "OPUS supports at most eight tensor dimensions");
    aiter_tensor_t result{};
    result.ptr = tensor.data_ptr();
    result.numel_ = tensor.numel();
    result.ndim = tensor.dim();
    result.device_id = tensor.is_cuda() ? tensor.get_device() : -1;
    for(int i = 0; i < tensor.dim(); ++i)
    {
        result.shape[i] = tensor.size(i);
        result.strides[i] = tensor.stride(i);
    }
    switch(tensor.scalar_type())
    {
    case at::ScalarType::Float8_e4m3fn: result.dtype_ = AITER_DTYPE_fp8; break;
    case at::ScalarType::Float8_e8m0fnu: result.dtype_ = AITER_DTYPE_fp8_e8m0; break;
    case at::ScalarType::BFloat16: result.dtype_ = AITER_DTYPE_bf16; break;
    case at::ScalarType::Float: result.dtype_ = AITER_DTYPE_fp32; break;
    case at::ScalarType::Byte: result.dtype_ = AITER_DTYPE_u8; break;
    default: TORCH_CHECK(false, "Unsupported OPUS BPreshuffle tensor dtype: ", tensor.scalar_type());
    }
    return result;
}

at::Tensor opus_bpreshuffle_tensor_launch(
    const at::Tensor& x, const at::Tensor& w,
    const at::Tensor& xs, const at::Tensor& ws, const at::Tensor& y,
    int kid, const std::optional<at::Tensor>& workspace)
{
    auto a = opus_tensor_descriptor(x);
    auto b = opus_tensor_descriptor(w);
    auto sa = opus_tensor_descriptor(xs);
    auto sb = opus_tensor_descriptor(ws);
    auto c = opus_tensor_descriptor(y);
    // Resolve the current stream at execution, including graph capture and
    // torch.compile. The checked launcher still verifies the current device.
    if(x.is_cuda())
        aiter::setCurrentHIPStream(c10::cuda::getCurrentCUDAStream(x.get_device()).stream());
    if(workspace)
    {
        auto temporary = opus_tensor_descriptor(*workspace);
        opus_gemm_a8w8_blockscale_bpreshuffle_workspace_launch(a, b, sa, sb, c, kid, temporary);
    }
    else
        opus_gemm_a8w8_blockscale_bpreshuffle_launch(a, b, sa, sb, c, kid);
    return y;
}
} // namespace

void opus_gemm_bind_tensor_launches(pybind11::module_& module)
{
    module.def("opus_gemm_a8w8_blockscale_bpreshuffle_tensor_launch",
        [](const at::Tensor& x, const at::Tensor& w, const at::Tensor& xs,
           const at::Tensor& ws, const at::Tensor& y, int kid) {
            return opus_bpreshuffle_tensor_launch(x, w, xs, ws, y, kid, std::nullopt);
        });
    module.def("opus_gemm_a8w8_blockscale_bpreshuffle_workspace_tensor_launch",
        [](const at::Tensor& x, const at::Tensor& w, const at::Tensor& xs,
           const at::Tensor& ws, const at::Tensor& y, int kid, const at::Tensor& workspace) {
            return opus_bpreshuffle_tensor_launch(x, w, xs, ws, y, kid, workspace);
        });
}

#endif // !__HIP_DEVICE_COMPILE__
