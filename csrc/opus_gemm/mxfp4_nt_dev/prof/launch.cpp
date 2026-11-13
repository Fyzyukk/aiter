// launch <co> M N K iters : random fp4 / scales, no verification (profiling only)
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include "gemm_a4w4_mxfp4_4wave_traits.hpp"
#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { printf("HIP %s @%d\n", hipGetErrorString(e), __LINE__); exit(1);} } while (0)
int main(int argc, char** argv) {
    const char* co = argv[1]; const int M = atoi(argv[2]), N = atoi(argv[3]), K = atoi(argv[4]), it = atoi(argv[5]); const char* sym = argv[6];
    std::vector<unsigned char> a(size_t(M) * K / 2), b(size_t(N) * K / 2), sa(size_t(M) * K / 32), sb(size_t(N) * K / 32);
    for (auto& v : a) v = rand(); for (auto& v : b) v = rand();
    for (auto& v : sa) v = 124 + rand() % 7; for (auto& v : sb) v = 124 + rand() % 7;
    void *da, *db, *dc, *dsa, *dsb;
    CK(hipMalloc(&da, a.size())); CK(hipMalloc(&db, b.size())); CK(hipMalloc(&dc, size_t(M) * N * 4));
    CK(hipMalloc(&dsa, sa.size())); CK(hipMalloc(&dsb, sb.size()));
    CK(hipMemcpy(da, a.data(), a.size(), hipMemcpyHostToDevice)); CK(hipMemcpy(db, b.data(), b.size(), hipMemcpyHostToDevice));
    CK(hipMemcpy(dsa, sa.data(), sa.size(), hipMemcpyHostToDevice)); CK(hipMemcpy(dsb, sb.data(), sb.size(), hipMemcpyHostToDevice));
    opus_gemm_mxfp4_scale_kargs k{};
    k.ptr_a = da; k.ptr_b = db; k.ptr_c = dc; k.m = M; k.n = N; k.k = K; k.batch = 1;
    k.stride_a = K / 2; k.stride_b = K / 2; k.stride_c = N; k.stride_a_batch = M * (K / 2); k.stride_b_batch = N * (K / 2); k.stride_c_batch = M * N;
    k.ptr_sfa = dsa; k.ptr_sfb = dsb; k.stride_sfa = K / 32; k.stride_sfb = K / 32; k.stride_sfa_batch = M * (K / 32); k.stride_sfb_batch = N * (K / 32);
    hipModule_t mod; hipFunction_t fn; CK(hipModuleLoad(&mod, co));
    CK(hipModuleGetFunction(&fn, mod, sym));
    size_t sz = sizeof(k); void* cfg[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &k, HIP_LAUNCH_PARAM_BUFFER_SIZE, &sz, HIP_LAUNCH_PARAM_END};
    int dev_cu = 0; CK(hipDeviceGetAttribute(&dev_cu, hipDeviceAttributeMultiprocessorCount, 0));
    const int grid = getenv("GRID") ? atoi(getenv("GRID")) : std::min((M / 256) * (N / 256), dev_cu); k.grid_blocks = grid;
    for (int i = 0; i < it; ++i) CK(hipModuleLaunchKernel(fn, grid, 1, 1, 256, 1, 1, 0, 0, nullptr, cfg));
    CK(hipDeviceSynchronize()); printf("done\n");
}
