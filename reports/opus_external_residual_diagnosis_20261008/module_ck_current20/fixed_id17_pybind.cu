#include "gemm_a8w8_blockscale_bpreshuffle_common.cuh"
#include "gemm_a8w8_blockscale_bpreshuffle_manifest.h"
PYBIND11_MODULE(AITER_EXTENSION_NAME,m) { m.def("gemm_a8w8_blockscale_bpreshuffle_tune", [](torch::Tensor& a,torch::Tensor& b,torch::Tensor& sa,torch::Tensor& sb,torch::Tensor& out,int kid,int split) { TORCH_CHECK(kid==17 && split==0,"fixed ID17 only"); return a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<F32,B16>(a,b,sa,sb,out); }); }
