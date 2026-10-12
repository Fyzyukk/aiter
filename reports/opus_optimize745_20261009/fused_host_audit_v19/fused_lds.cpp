#include <cassert>
#include <cstdint>
#include <cstdio>
#include "/root/workspace/aiter-opus-mxfp8-bpreshuffle/reports/opus_optimize745_20261009/fused_lds/fused_kargs.cuh"
using FusedTraits0 = opus_gemm_mxscale_bpreshuffle_pipeline_traits<opus_gemm_mxscale_bpreshuffle_runtime_config_traits<opus_gemm_small_lds_traits_gfx950<48, 64, 1, 4, 3, 1, 2, false, false, true, true, true, 1, 16, 128, 2, 0, true>, 16, 128>,1>;
using FusedTraits1 = opus_gemm_mxscale_bpreshuffle_pipeline_traits<opus_gemm_mxscale_bpreshuffle_runtime_config_traits<opus_gemm_small_lds_traits_gfx950<48, 64, 1, 4, 4, 1, 2, false, false, true, true, true, 1, 16, 128, 2, 0, true>, 16, 128>,1>;
using FusedTraits2 = opus_gemm_mxscale_bpreshuffle_pipeline_traits<opus_gemm_mxscale_bpreshuffle_runtime_config_traits<opus_gemm_small_lds_traits_gfx950<64, 64, 2, 2, 3, 1, 2, false, false, true, true, true, 1, 16, 128, 2, 0, true>, 16, 128>,1>;
using FusedTraits3 = opus_gemm_mxscale_bpreshuffle_pipeline_traits<opus_gemm_mxscale_bpreshuffle_runtime_config_traits<opus_gemm_small_lds_traits_gfx950<64, 64, 2, 2, 4, 1, 2, false, false, true, true, true, 1, 16, 128, 2, 0, true>, 16, 128>,1>;
using FusedTraits4 = opus_gemm_mxscale_bpreshuffle_pipeline_traits<opus_gemm_mxscale_bpreshuffle_runtime_config_traits<opus_gemm_small_lds_traits_gfx950<64, 128, 2, 2, 3, 1, 2, false, false, true, true, true, 1, 16, 128, 2, 0, true>, 16, 128>,1>;
using FusedTraits5 = opus_gemm_mxscale_bpreshuffle_pipeline_traits<opus_gemm_mxscale_bpreshuffle_runtime_config_traits<opus_gemm_small_lds_traits_gfx950<64, 128, 2, 2, 4, 1, 2, false, false, true, true, true, 1, 16, 128, 2, 0, true>, 16, 128>,1>;
void fused_lds_kernel0(opus745_fused_lds_kargs) {}
void fused_lds_kernel1(opus745_fused_lds_kargs) {}
void fused_lds_kernel2(opus745_fused_lds_kargs) {}
void fused_lds_kernel3(opus745_fused_lds_kargs) {}
void fused_lds_kernel4(opus745_fused_lds_kargs) {}
void fused_lds_kernel5(opus745_fused_lds_kargs) {}
#if !defined(__HIP_DEVICE_COMPILE__)
template<class T> int launch_one(opus745_fused_lds_kargs args, void* stream, const void* kernel) {
    const int tiles = ((args.m + T::B_M - 1) / T::B_M) * (args.n / T::B_N);
    const dim3 grid(((tiles + 7) / 8) * 8 * args.split_k);
    void* argv[] = {&args};
    return hipLaunchKernel(kernel,grid,dim3(T::BLOCK_SIZE),argv,T::lds_bytes(args.k,args.split_k),stream);
}
extern "C" __attribute__((visibility("default")))
int launch(int variant,const void* a,const void* b,const void* sfa,const void* sfb,
           void* c,float* workspace,int* counters,int m,int n,int k,int split_k,void* stream) {
    if (m<=0 || m>2048 || n<=0 || k<128 || k>16384 || n%128 || k%128 ||
        split_k<1 || split_k>16 || split_k>k/128 || int64_t(m)*k>INT32_MAX ||
        int64_t(n)*k>INT32_MAX || int64_t(m)*n*2>INT32_MAX ||
        !a || !b || !sfa || !sfb || !c || (split_k>1 && (!workspace || !counters)) ||
        reinterpret_cast<uintptr_t>(a)%16 || reinterpret_cast<uintptr_t>(b)%16 ||
        reinterpret_cast<uintptr_t>(sfa)%16 || reinterpret_cast<uintptr_t>(sfb)%4 ||
        reinterpret_cast<uintptr_t>(c)%16 || reinterpret_cast<uintptr_t>(workspace)%16 ||
        reinterpret_cast<uintptr_t>(counters)%4) return 1;
    opus745_fused_lds_kargs args{};
    args.ptr_a=a;args.ptr_b=b;args.ptr_sfa=sfa;args.ptr_sfb=sfb;args.ptr_c=c;
    args.ptr_workspace=workspace;args.ptr_counters=counters;args.split_k=split_k;
    args.m=m;args.n=n;args.k=k;args.batch=1;
    args.stride_a=k;args.stride_b=k;args.stride_c=n;args.stride_sfa=m;args.stride_sfb=k/128;
    switch (variant) {

    case 0:return launch_one<FusedTraits0>(args,stream,reinterpret_cast<const void*>(fused_lds_kernel0));
    case 1:return launch_one<FusedTraits1>(args,stream,reinterpret_cast<const void*>(fused_lds_kernel1));
    case 2:return launch_one<FusedTraits2>(args,stream,reinterpret_cast<const void*>(fused_lds_kernel2));
    case 3:return launch_one<FusedTraits3>(args,stream,reinterpret_cast<const void*>(fused_lds_kernel3));
    case 4:return launch_one<FusedTraits4>(args,stream,reinterpret_cast<const void*>(fused_lds_kernel4));
    case 5:return launch_one<FusedTraits5>(args,stream,reinterpret_cast<const void*>(fused_lds_kernel5));
    default:return 1;
    }
}
#endif

static_assert(sizeof(opus745_fused_lds_kargs) == 120);
static opus745_fused_lds_kargs recorded;
static dim3 grid, block;
static size_t dynamic_lds; static int calls; static const void* recorded_kernel;
extern "C" int hipLaunchKernel(const void* kernel, dim3 g, dim3 b,
    void** argv, size_t lds, void* stream) {
    assert(stream == reinterpret_cast<void*>(0x8000));
    grid=g;block=b;dynamic_lds=lds;recorded_kernel=kernel;calls++;

    recorded=*static_cast<opus745_fused_lds_kargs*>(argv[0]);return 0;
}
static int call(int variant, int m, int n, int k, int split,
    uintptr_t a=0x1000, uintptr_t b=0x2000, uintptr_t sa=0x3000,
    uintptr_t sb=0x4000, uintptr_t c=0x5000, uintptr_t ws=0x6000, uintptr_t counters=0x7000) {
    return launch(variant,reinterpret_cast<void*>(a),reinterpret_cast<void*>(b),
      reinterpret_cast<void*>(sa),reinterpret_cast<void*>(sb),reinterpret_cast<void*>(c),
      reinterpret_cast<float*>(ws),reinterpret_cast<int*>(counters),m,n,k,split,
      reinterpret_cast<void*>(0x8000));
}
int main() { int valid=0, rejected=0;

{ int before=calls;assert(call(0,1,128,384,3,
    0x1000,0x2000,0x3000,0x4000,0x5000,24576,28672)==0);assert(calls==before+1);
    assert(grid.x==24 && grid.y==1 && grid.z==1);
    assert(block.x==256 && block.y==1 && block.z==1 && dynamic_lds==14833);
    assert(recorded_kernel==reinterpret_cast<const void*>(fused_lds_kernel0));
    assert(recorded.m==1 && recorded.n==128 && recorded.k==384 && recorded.batch==1);
    assert(recorded.split_k==3 && recorded.stride_a==384 && recorded.stride_b==384);
    assert(recorded.stride_c==128 && recorded.stride_sfa==1 && recorded.stride_sfb==3);
    assert(recorded.ptr_a==reinterpret_cast<void*>(0x1000));
    assert(recorded.ptr_b==reinterpret_cast<void*>(0x2000));
    assert(recorded.ptr_sfa==reinterpret_cast<void*>(0x3000));
    assert(recorded.ptr_sfb==reinterpret_cast<void*>(0x4000));
    assert(recorded.ptr_c==reinterpret_cast<void*>(0x5000));
    assert(recorded.ptr_workspace==reinterpret_cast<float*>(24576));
    assert(recorded.ptr_counters==reinterpret_cast<int*>(28672));valid++; }
{ int before=calls;assert(call(0,17,256,7168,4,
    0x1000,0x2000,0x3000,0x4000,0x5000,24576,28672)==0);assert(calls==before+1);
    assert(grid.x==32 && grid.y==1 && grid.z==1);
    assert(block.x==256 && block.y==1 && block.z==1 && dynamic_lds==45038);
    assert(recorded_kernel==reinterpret_cast<const void*>(fused_lds_kernel0));
    assert(recorded.m==17 && recorded.n==256 && recorded.k==7168 && recorded.batch==1);
    assert(recorded.split_k==4 && recorded.stride_a==7168 && recorded.stride_b==7168);
    assert(recorded.stride_c==256 && recorded.stride_sfa==17 && recorded.stride_sfb==56);
    assert(recorded.ptr_a==reinterpret_cast<void*>(0x1000));
    assert(recorded.ptr_b==reinterpret_cast<void*>(0x2000));
    assert(recorded.ptr_sfa==reinterpret_cast<void*>(0x3000));
    assert(recorded.ptr_sfb==reinterpret_cast<void*>(0x4000));
    assert(recorded.ptr_c==reinterpret_cast<void*>(0x5000));
    assert(recorded.ptr_workspace==reinterpret_cast<float*>(24576));
    assert(recorded.ptr_counters==reinterpret_cast<int*>(28672));valid++; }
{ int before=calls;assert(call(0,2048,128,16384,16,
    0x1000,0x2000,0x3000,0x4000,0x5000,24576,28672)==0);assert(calls==before+1);
    assert(grid.x==1408 && grid.y==1 && grid.z==1);
    assert(block.x==256 && block.y==1 && block.z==1 && dynamic_lds==44744);
    assert(recorded_kernel==reinterpret_cast<const void*>(fused_lds_kernel0));
    assert(recorded.m==2048 && recorded.n==128 && recorded.k==16384 && recorded.batch==1);
    assert(recorded.split_k==16 && recorded.stride_a==16384 && recorded.stride_b==16384);
    assert(recorded.stride_c==128 && recorded.stride_sfa==2048 && recorded.stride_sfb==128);
    assert(recorded.ptr_a==reinterpret_cast<void*>(0x1000));
    assert(recorded.ptr_b==reinterpret_cast<void*>(0x2000));
    assert(recorded.ptr_sfa==reinterpret_cast<void*>(0x3000));
    assert(recorded.ptr_sfb==reinterpret_cast<void*>(0x4000));
    assert(recorded.ptr_c==reinterpret_cast<void*>(0x5000));
    assert(recorded.ptr_workspace==reinterpret_cast<float*>(24576));
    assert(recorded.ptr_counters==reinterpret_cast<int*>(28672));valid++; }
{ int before=calls;assert(call(0,16,128,128,1,
    0x1000,0x2000,0x3000,0x4000,0x5000,0,0)==0);assert(calls==before+1);
    assert(grid.x==8 && grid.y==1 && grid.z==1);
    assert(block.x==256 && block.y==1 && block.z==1 && dynamic_lds==14833);
    assert(recorded_kernel==reinterpret_cast<const void*>(fused_lds_kernel0));
    assert(recorded.m==16 && recorded.n==128 && recorded.k==128 && recorded.batch==1);
    assert(recorded.split_k==1 && recorded.stride_a==128 && recorded.stride_b==128);
    assert(recorded.stride_c==128 && recorded.stride_sfa==16 && recorded.stride_sfb==1);
    assert(recorded.ptr_a==reinterpret_cast<void*>(0x1000));
    assert(recorded.ptr_b==reinterpret_cast<void*>(0x2000));
    assert(recorded.ptr_sfa==reinterpret_cast<void*>(0x3000));
    assert(recorded.ptr_sfb==reinterpret_cast<void*>(0x4000));
    assert(recorded.ptr_c==reinterpret_cast<void*>(0x5000));
    assert(recorded.ptr_workspace==reinterpret_cast<float*>(0));
    assert(recorded.ptr_counters==reinterpret_cast<int*>(0));valid++; }
{int before=calls;assert(call(0,0,128,384,3)==1);assert(calls==before);rejected++;}
{int before=calls;assert(call(0,2049,128,384,3)==1);assert(calls==before);rejected++;}
{int before=calls;assert(call(0,16,127,384,3)==1);assert(calls==before);rejected++;}
{int before=calls;assert(call(0,16,128,129,1)==1);assert(calls==before);rejected++;}
{int before=calls;assert(call(0,16,128,384,0)==1);assert(calls==before);rejected++;}
{int before=calls;assert(call(0,16,128,384,4)==1);assert(calls==before);rejected++;}
{int before=calls;assert(call(0,16,128,384,3,0x0)==1);assert(calls==before);rejected++;}
{int before=calls;assert(call(0,16,128,384,3,0x1001)==1);assert(calls==before);rejected++;}
{int before=calls;assert(call(0,16,128,384,3,0x1000,0x2000,0x3000,0x4000,0x5000,0x0)==1);assert(calls==before);rejected++;}
{int before=calls;assert(call(0,16,128,384,3,0x1000,0x2000,0x3000,0x4000,0x5000,0x6000,0x0)==1);assert(calls==before);rejected++;}
{int before=calls;assert(call(0,16,128,384,3,0x1000,0x2000,0x3000,0x4000,0x5000,0x6001,0x7000)==1);assert(calls==before);rejected++;}
{int before=calls;assert(call(0,16,128,384,3,0x1000,0x2000,0x3000,0x4000,0x5000,0x6000,0x7001)==1);assert(calls==before);rejected++;}
{int before=calls;assert(call(0,16,128,384,3,0x1000,0x2000,0x3001)==1);assert(calls==before);rejected++;}
{int before=calls;assert(call(0,16,128,384,3,0x1000,0x2000,0x3000,0x4001)==1);assert(calls==before);rejected++;}
{ int before=calls;assert(call(1,1,128,384,3,
    0x1000,0x2000,0x3000,0x4000,0x5000,24576,28672)==0);assert(calls==before+1);
    assert(grid.x==24 && grid.y==1 && grid.z==1);
    assert(block.x==256 && block.y==1 && block.z==1 && dynamic_lds==14833);
    assert(recorded_kernel==reinterpret_cast<const void*>(fused_lds_kernel1));
    assert(recorded.m==1 && recorded.n==128 && recorded.k==384 && recorded.batch==1);
    assert(recorded.split_k==3 && recorded.stride_a==384 && recorded.stride_b==384);
    assert(recorded.stride_c==128 && recorded.stride_sfa==1 && recorded.stride_sfb==3);
    assert(recorded.ptr_a==reinterpret_cast<void*>(0x1000));
    assert(recorded.ptr_b==reinterpret_cast<void*>(0x2000));
    assert(recorded.ptr_sfa==reinterpret_cast<void*>(0x3000));
    assert(recorded.ptr_sfb==reinterpret_cast<void*>(0x4000));
    assert(recorded.ptr_c==reinterpret_cast<void*>(0x5000));
    assert(recorded.ptr_workspace==reinterpret_cast<float*>(24576));
    assert(recorded.ptr_counters==reinterpret_cast<int*>(28672));valid++; }
{ int before=calls;assert(call(1,17,256,7168,4,
    0x1000,0x2000,0x3000,0x4000,0x5000,24576,28672)==0);assert(calls==before+1);
    assert(grid.x==32 && grid.y==1 && grid.z==1);
    assert(block.x==256 && block.y==1 && block.z==1 && dynamic_lds==59822);
    assert(recorded_kernel==reinterpret_cast<const void*>(fused_lds_kernel1));
    assert(recorded.m==17 && recorded.n==256 && recorded.k==7168 && recorded.batch==1);
    assert(recorded.split_k==4 && recorded.stride_a==7168 && recorded.stride_b==7168);
    assert(recorded.stride_c==256 && recorded.stride_sfa==17 && recorded.stride_sfb==56);
    assert(recorded.ptr_a==reinterpret_cast<void*>(0x1000));
    assert(recorded.ptr_b==reinterpret_cast<void*>(0x2000));
    assert(recorded.ptr_sfa==reinterpret_cast<void*>(0x3000));
    assert(recorded.ptr_sfb==reinterpret_cast<void*>(0x4000));
    assert(recorded.ptr_c==reinterpret_cast<void*>(0x5000));
    assert(recorded.ptr_workspace==reinterpret_cast<float*>(24576));
    assert(recorded.ptr_counters==reinterpret_cast<int*>(28672));valid++; }
{ int before=calls;assert(call(1,2048,128,16384,16,
    0x1000,0x2000,0x3000,0x4000,0x5000,24576,28672)==0);assert(calls==before+1);
    assert(grid.x==1408 && grid.y==1 && grid.z==1);
    assert(block.x==256 && block.y==1 && block.z==1 && dynamic_lds==59528);
    assert(recorded_kernel==reinterpret_cast<const void*>(fused_lds_kernel1));
    assert(recorded.m==2048 && recorded.n==128 && recorded.k==16384 && recorded.batch==1);
    assert(recorded.split_k==16 && recorded.stride_a==16384 && recorded.stride_b==16384);
    assert(recorded.stride_c==128 && recorded.stride_sfa==2048 && recorded.stride_sfb==128);
    assert(recorded.ptr_a==reinterpret_cast<void*>(0x1000));
    assert(recorded.ptr_b==reinterpret_cast<void*>(0x2000));
    assert(recorded.ptr_sfa==reinterpret_cast<void*>(0x3000));
    assert(recorded.ptr_sfb==reinterpret_cast<void*>(0x4000));
    assert(recorded.ptr_c==reinterpret_cast<void*>(0x5000));
    assert(recorded.ptr_workspace==reinterpret_cast<float*>(24576));
    assert(recorded.ptr_counters==reinterpret_cast<int*>(28672));valid++; }
{ int before=calls;assert(call(1,16,128,128,1,
    0x1000,0x2000,0x3000,0x4000,0x5000,0,0)==0);assert(calls==before+1);
    assert(grid.x==8 && grid.y==1 && grid.z==1);
    assert(block.x==256 && block.y==1 && block.z==1 && dynamic_lds==14833);
    assert(recorded_kernel==reinterpret_cast<const void*>(fused_lds_kernel1));
    assert(recorded.m==16 && recorded.n==128 && recorded.k==128 && recorded.batch==1);
    assert(recorded.split_k==1 && recorded.stride_a==128 && recorded.stride_b==128);
    assert(recorded.stride_c==128 && recorded.stride_sfa==16 && recorded.stride_sfb==1);
    assert(recorded.ptr_a==reinterpret_cast<void*>(0x1000));
    assert(recorded.ptr_b==reinterpret_cast<void*>(0x2000));
    assert(recorded.ptr_sfa==reinterpret_cast<void*>(0x3000));
    assert(recorded.ptr_sfb==reinterpret_cast<void*>(0x4000));
    assert(recorded.ptr_c==reinterpret_cast<void*>(0x5000));
    assert(recorded.ptr_workspace==reinterpret_cast<float*>(0));
    assert(recorded.ptr_counters==reinterpret_cast<int*>(0));valid++; }
{int before=calls;assert(call(1,0,128,384,3)==1);assert(calls==before);rejected++;}
{int before=calls;assert(call(1,2049,128,384,3)==1);assert(calls==before);rejected++;}
{int before=calls;assert(call(1,16,127,384,3)==1);assert(calls==before);rejected++;}
{int before=calls;assert(call(1,16,128,129,1)==1);assert(calls==before);rejected++;}
{int before=calls;assert(call(1,16,128,384,0)==1);assert(calls==before);rejected++;}
{int before=calls;assert(call(1,16,128,384,4)==1);assert(calls==before);rejected++;}
{int before=calls;assert(call(1,16,128,384,3,0x0)==1);assert(calls==before);rejected++;}
{int before=calls;assert(call(1,16,128,384,3,0x1001)==1);assert(calls==before);rejected++;}
{int before=calls;assert(call(1,16,128,384,3,0x1000,0x2000,0x3000,0x4000,0x5000,0x0)==1);assert(calls==before);rejected++;}
{int before=calls;assert(call(1,16,128,384,3,0x1000,0x2000,0x3000,0x4000,0x5000,0x6000,0x0)==1);assert(calls==before);rejected++;}
{int before=calls;assert(call(1,16,128,384,3,0x1000,0x2000,0x3000,0x4000,0x5000,0x6001,0x7000)==1);assert(calls==before);rejected++;}
{int before=calls;assert(call(1,16,128,384,3,0x1000,0x2000,0x3000,0x4000,0x5000,0x6000,0x7001)==1);assert(calls==before);rejected++;}
{int before=calls;assert(call(1,16,128,384,3,0x1000,0x2000,0x3001)==1);assert(calls==before);rejected++;}
{int before=calls;assert(call(1,16,128,384,3,0x1000,0x2000,0x3000,0x4001)==1);assert(calls==before);rejected++;}
{ int before=calls;assert(call(2,1,128,384,3,
    0x1000,0x2000,0x3000,0x4000,0x5000,24576,28672)==0);assert(calls==before+1);
    assert(grid.x==24 && grid.y==1 && grid.z==1);
    assert(block.x==256 && block.y==1 && block.z==1 && dynamic_lds==16961);
    assert(recorded_kernel==reinterpret_cast<const void*>(fused_lds_kernel2));
    assert(recorded.m==1 && recorded.n==128 && recorded.k==384 && recorded.batch==1);
    assert(recorded.split_k==3 && recorded.stride_a==384 && recorded.stride_b==384);
    assert(recorded.stride_c==128 && recorded.stride_sfa==1 && recorded.stride_sfb==3);
    assert(recorded.ptr_a==reinterpret_cast<void*>(0x1000));
    assert(recorded.ptr_b==reinterpret_cast<void*>(0x2000));
    assert(recorded.ptr_sfa==reinterpret_cast<void*>(0x3000));
    assert(recorded.ptr_sfb==reinterpret_cast<void*>(0x4000));
    assert(recorded.ptr_c==reinterpret_cast<void*>(0x5000));
    assert(recorded.ptr_workspace==reinterpret_cast<float*>(24576));
    assert(recorded.ptr_counters==reinterpret_cast<int*>(28672));valid++; }
{ int before=calls;assert(call(2,17,256,7168,4,
    0x1000,0x2000,0x3000,0x4000,0x5000,24576,28672)==0);assert(calls==before+1);
    assert(grid.x==32 && grid.y==1 && grid.z==1);
    assert(block.x==256 && block.y==1 && block.z==1 && dynamic_lds==51598);
    assert(recorded_kernel==reinterpret_cast<const void*>(fused_lds_kernel2));
    assert(recorded.m==17 && recorded.n==256 && recorded.k==7168 && recorded.batch==1);
    assert(recorded.split_k==4 && recorded.stride_a==7168 && recorded.stride_b==7168);
    assert(recorded.stride_c==256 && recorded.stride_sfa==17 && recorded.stride_sfb==56);
    assert(recorded.ptr_a==reinterpret_cast<void*>(0x1000));
    assert(recorded.ptr_b==reinterpret_cast<void*>(0x2000));
    assert(recorded.ptr_sfa==reinterpret_cast<void*>(0x3000));
    assert(recorded.ptr_sfb==reinterpret_cast<void*>(0x4000));
    assert(recorded.ptr_c==reinterpret_cast<void*>(0x5000));
    assert(recorded.ptr_workspace==reinterpret_cast<float*>(24576));
    assert(recorded.ptr_counters==reinterpret_cast<int*>(28672));valid++; }
{ int before=calls;assert(call(2,2048,128,16384,16,
    0x1000,0x2000,0x3000,0x4000,0x5000,24576,28672)==0);assert(calls==before+1);
    assert(grid.x==1024 && grid.y==1 && grid.z==1);
    assert(block.x==256 && block.y==1 && block.z==1 && dynamic_lds==51208);
    assert(recorded_kernel==reinterpret_cast<const void*>(fused_lds_kernel2));
    assert(recorded.m==2048 && recorded.n==128 && recorded.k==16384 && recorded.batch==1);
    assert(recorded.split_k==16 && recorded.stride_a==16384 && recorded.stride_b==16384);
    assert(recorded.stride_c==128 && recorded.stride_sfa==2048 && recorded.stride_sfb==128);
    assert(recorded.ptr_a==reinterpret_cast<void*>(0x1000));
    assert(recorded.ptr_b==reinterpret_cast<void*>(0x2000));
    assert(recorded.ptr_sfa==reinterpret_cast<void*>(0x3000));
    assert(recorded.ptr_sfb==reinterpret_cast<void*>(0x4000));
    assert(recorded.ptr_c==reinterpret_cast<void*>(0x5000));
    assert(recorded.ptr_workspace==reinterpret_cast<float*>(24576));
    assert(recorded.ptr_counters==reinterpret_cast<int*>(28672));valid++; }
{ int before=calls;assert(call(2,16,128,128,1,
    0x1000,0x2000,0x3000,0x4000,0x5000,0,0)==0);assert(calls==before+1);
    assert(grid.x==8 && grid.y==1 && grid.z==1);
    assert(block.x==256 && block.y==1 && block.z==1 && dynamic_lds==16961);
    assert(recorded_kernel==reinterpret_cast<const void*>(fused_lds_kernel2));
    assert(recorded.m==16 && recorded.n==128 && recorded.k==128 && recorded.batch==1);
    assert(recorded.split_k==1 && recorded.stride_a==128 && recorded.stride_b==128);
    assert(recorded.stride_c==128 && recorded.stride_sfa==16 && recorded.stride_sfb==1);
    assert(recorded.ptr_a==reinterpret_cast<void*>(0x1000));
    assert(recorded.ptr_b==reinterpret_cast<void*>(0x2000));
    assert(recorded.ptr_sfa==reinterpret_cast<void*>(0x3000));
    assert(recorded.ptr_sfb==reinterpret_cast<void*>(0x4000));
    assert(recorded.ptr_c==reinterpret_cast<void*>(0x5000));
    assert(recorded.ptr_workspace==reinterpret_cast<float*>(0));
    assert(recorded.ptr_counters==reinterpret_cast<int*>(0));valid++; }
{int before=calls;assert(call(2,0,128,384,3)==1);assert(calls==before);rejected++;}
{int before=calls;assert(call(2,2049,128,384,3)==1);assert(calls==before);rejected++;}
{int before=calls;assert(call(2,16,127,384,3)==1);assert(calls==before);rejected++;}
{int before=calls;assert(call(2,16,128,129,1)==1);assert(calls==before);rejected++;}
{int before=calls;assert(call(2,16,128,384,0)==1);assert(calls==before);rejected++;}
{int before=calls;assert(call(2,16,128,384,4)==1);assert(calls==before);rejected++;}
{int before=calls;assert(call(2,16,128,384,3,0x0)==1);assert(calls==before);rejected++;}
{int before=calls;assert(call(2,16,128,384,3,0x1001)==1);assert(calls==before);rejected++;}
{int before=calls;assert(call(2,16,128,384,3,0x1000,0x2000,0x3000,0x4000,0x5000,0x0)==1);assert(calls==before);rejected++;}
{int before=calls;assert(call(2,16,128,384,3,0x1000,0x2000,0x3000,0x4000,0x5000,0x6000,0x0)==1);assert(calls==before);rejected++;}
{int before=calls;assert(call(2,16,128,384,3,0x1000,0x2000,0x3000,0x4000,0x5000,0x6001,0x7000)==1);assert(calls==before);rejected++;}
{int before=calls;assert(call(2,16,128,384,3,0x1000,0x2000,0x3000,0x4000,0x5000,0x6000,0x7001)==1);assert(calls==before);rejected++;}
{int before=calls;assert(call(2,16,128,384,3,0x1000,0x2000,0x3001)==1);assert(calls==before);rejected++;}
{int before=calls;assert(call(2,16,128,384,3,0x1000,0x2000,0x3000,0x4001)==1);assert(calls==before);rejected++;}
{ int before=calls;assert(call(3,1,128,384,3,
    0x1000,0x2000,0x3000,0x4000,0x5000,24576,28672)==0);assert(calls==before+1);
    assert(grid.x==24 && grid.y==1 && grid.z==1);
    assert(block.x==256 && block.y==1 && block.z==1 && dynamic_lds==16961);
    assert(recorded_kernel==reinterpret_cast<const void*>(fused_lds_kernel3));
    assert(recorded.m==1 && recorded.n==128 && recorded.k==384 && recorded.batch==1);
    assert(recorded.split_k==3 && recorded.stride_a==384 && recorded.stride_b==384);
    assert(recorded.stride_c==128 && recorded.stride_sfa==1 && recorded.stride_sfb==3);
    assert(recorded.ptr_a==reinterpret_cast<void*>(0x1000));
    assert(recorded.ptr_b==reinterpret_cast<void*>(0x2000));
    assert(recorded.ptr_sfa==reinterpret_cast<void*>(0x3000));
    assert(recorded.ptr_sfb==reinterpret_cast<void*>(0x4000));
    assert(recorded.ptr_c==reinterpret_cast<void*>(0x5000));
    assert(recorded.ptr_workspace==reinterpret_cast<float*>(24576));
    assert(recorded.ptr_counters==reinterpret_cast<int*>(28672));valid++; }
{ int before=calls;assert(call(3,17,256,7168,4,
    0x1000,0x2000,0x3000,0x4000,0x5000,24576,28672)==0);assert(calls==before+1);
    assert(grid.x==32 && grid.y==1 && grid.z==1);
    assert(block.x==256 && block.y==1 && block.z==1 && dynamic_lds==68494);
    assert(recorded_kernel==reinterpret_cast<const void*>(fused_lds_kernel3));
    assert(recorded.m==17 && recorded.n==256 && recorded.k==7168 && recorded.batch==1);
    assert(recorded.split_k==4 && recorded.stride_a==7168 && recorded.stride_b==7168);
    assert(recorded.stride_c==256 && recorded.stride_sfa==17 && recorded.stride_sfb==56);
    assert(recorded.ptr_a==reinterpret_cast<void*>(0x1000));
    assert(recorded.ptr_b==reinterpret_cast<void*>(0x2000));
    assert(recorded.ptr_sfa==reinterpret_cast<void*>(0x3000));
    assert(recorded.ptr_sfb==reinterpret_cast<void*>(0x4000));
    assert(recorded.ptr_c==reinterpret_cast<void*>(0x5000));
    assert(recorded.ptr_workspace==reinterpret_cast<float*>(24576));
    assert(recorded.ptr_counters==reinterpret_cast<int*>(28672));valid++; }
{ int before=calls;assert(call(3,2048,128,16384,16,
    0x1000,0x2000,0x3000,0x4000,0x5000,24576,28672)==0);assert(calls==before+1);
    assert(grid.x==1024 && grid.y==1 && grid.z==1);
    assert(block.x==256 && block.y==1 && block.z==1 && dynamic_lds==68104);
    assert(recorded_kernel==reinterpret_cast<const void*>(fused_lds_kernel3));
    assert(recorded.m==2048 && recorded.n==128 && recorded.k==16384 && recorded.batch==1);
    assert(recorded.split_k==16 && recorded.stride_a==16384 && recorded.stride_b==16384);
    assert(recorded.stride_c==128 && recorded.stride_sfa==2048 && recorded.stride_sfb==128);
    assert(recorded.ptr_a==reinterpret_cast<void*>(0x1000));
    assert(recorded.ptr_b==reinterpret_cast<void*>(0x2000));
    assert(recorded.ptr_sfa==reinterpret_cast<void*>(0x3000));
    assert(recorded.ptr_sfb==reinterpret_cast<void*>(0x4000));
    assert(recorded.ptr_c==reinterpret_cast<void*>(0x5000));
    assert(recorded.ptr_workspace==reinterpret_cast<float*>(24576));
    assert(recorded.ptr_counters==reinterpret_cast<int*>(28672));valid++; }
{ int before=calls;assert(call(3,16,128,128,1,
    0x1000,0x2000,0x3000,0x4000,0x5000,0,0)==0);assert(calls==before+1);
    assert(grid.x==8 && grid.y==1 && grid.z==1);
    assert(block.x==256 && block.y==1 && block.z==1 && dynamic_lds==16961);
    assert(recorded_kernel==reinterpret_cast<const void*>(fused_lds_kernel3));
    assert(recorded.m==16 && recorded.n==128 && recorded.k==128 && recorded.batch==1);
    assert(recorded.split_k==1 && recorded.stride_a==128 && recorded.stride_b==128);
    assert(recorded.stride_c==128 && recorded.stride_sfa==16 && recorded.stride_sfb==1);
    assert(recorded.ptr_a==reinterpret_cast<void*>(0x1000));
    assert(recorded.ptr_b==reinterpret_cast<void*>(0x2000));
    assert(recorded.ptr_sfa==reinterpret_cast<void*>(0x3000));
    assert(recorded.ptr_sfb==reinterpret_cast<void*>(0x4000));
    assert(recorded.ptr_c==reinterpret_cast<void*>(0x5000));
    assert(recorded.ptr_workspace==reinterpret_cast<float*>(0));
    assert(recorded.ptr_counters==reinterpret_cast<int*>(0));valid++; }
{int before=calls;assert(call(3,0,128,384,3)==1);assert(calls==before);rejected++;}
{int before=calls;assert(call(3,2049,128,384,3)==1);assert(calls==before);rejected++;}
{int before=calls;assert(call(3,16,127,384,3)==1);assert(calls==before);rejected++;}
{int before=calls;assert(call(3,16,128,129,1)==1);assert(calls==before);rejected++;}
{int before=calls;assert(call(3,16,128,384,0)==1);assert(calls==before);rejected++;}
{int before=calls;assert(call(3,16,128,384,4)==1);assert(calls==before);rejected++;}
{int before=calls;assert(call(3,16,128,384,3,0x0)==1);assert(calls==before);rejected++;}
{int before=calls;assert(call(3,16,128,384,3,0x1001)==1);assert(calls==before);rejected++;}
{int before=calls;assert(call(3,16,128,384,3,0x1000,0x2000,0x3000,0x4000,0x5000,0x0)==1);assert(calls==before);rejected++;}
{int before=calls;assert(call(3,16,128,384,3,0x1000,0x2000,0x3000,0x4000,0x5000,0x6000,0x0)==1);assert(calls==before);rejected++;}
{int before=calls;assert(call(3,16,128,384,3,0x1000,0x2000,0x3000,0x4000,0x5000,0x6001,0x7000)==1);assert(calls==before);rejected++;}
{int before=calls;assert(call(3,16,128,384,3,0x1000,0x2000,0x3000,0x4000,0x5000,0x6000,0x7001)==1);assert(calls==before);rejected++;}
{int before=calls;assert(call(3,16,128,384,3,0x1000,0x2000,0x3001)==1);assert(calls==before);rejected++;}
{int before=calls;assert(call(3,16,128,384,3,0x1000,0x2000,0x3000,0x4001)==1);assert(calls==before);rejected++;}
{ int before=calls;assert(call(4,1,128,384,3,
    0x1000,0x2000,0x3000,0x4000,0x5000,24576,28672)==0);assert(calls==before+1);
    assert(grid.x==24 && grid.y==1 && grid.z==1);
    assert(block.x==256 && block.y==1 && block.z==1 && dynamic_lds==25409);
    assert(recorded_kernel==reinterpret_cast<const void*>(fused_lds_kernel4));
    assert(recorded.m==1 && recorded.n==128 && recorded.k==384 && recorded.batch==1);
    assert(recorded.split_k==3 && recorded.stride_a==384 && recorded.stride_b==384);
    assert(recorded.stride_c==128 && recorded.stride_sfa==1 && recorded.stride_sfb==3);
    assert(recorded.ptr_a==reinterpret_cast<void*>(0x1000));
    assert(recorded.ptr_b==reinterpret_cast<void*>(0x2000));
    assert(recorded.ptr_sfa==reinterpret_cast<void*>(0x3000));
    assert(recorded.ptr_sfb==reinterpret_cast<void*>(0x4000));
    assert(recorded.ptr_c==reinterpret_cast<void*>(0x5000));
    assert(recorded.ptr_workspace==reinterpret_cast<float*>(24576));
    assert(recorded.ptr_counters==reinterpret_cast<int*>(28672));valid++; }
{ int before=calls;assert(call(4,17,256,7168,4,
    0x1000,0x2000,0x3000,0x4000,0x5000,24576,28672)==0);assert(calls==before+1);
    assert(grid.x==32 && grid.y==1 && grid.z==1);
    assert(block.x==256 && block.y==1 && block.z==1 && dynamic_lds==76942);
    assert(recorded_kernel==reinterpret_cast<const void*>(fused_lds_kernel4));
    assert(recorded.m==17 && recorded.n==256 && recorded.k==7168 && recorded.batch==1);
    assert(recorded.split_k==4 && recorded.stride_a==7168 && recorded.stride_b==7168);
    assert(recorded.stride_c==256 && recorded.stride_sfa==17 && recorded.stride_sfb==56);
    assert(recorded.ptr_a==reinterpret_cast<void*>(0x1000));
    assert(recorded.ptr_b==reinterpret_cast<void*>(0x2000));
    assert(recorded.ptr_sfa==reinterpret_cast<void*>(0x3000));
    assert(recorded.ptr_sfb==reinterpret_cast<void*>(0x4000));
    assert(recorded.ptr_c==reinterpret_cast<void*>(0x5000));
    assert(recorded.ptr_workspace==reinterpret_cast<float*>(24576));
    assert(recorded.ptr_counters==reinterpret_cast<int*>(28672));valid++; }
{ int before=calls;assert(call(4,2048,128,16384,16,
    0x1000,0x2000,0x3000,0x4000,0x5000,24576,28672)==0);assert(calls==before+1);
    assert(grid.x==512 && grid.y==1 && grid.z==1);
    assert(block.x==256 && block.y==1 && block.z==1 && dynamic_lds==76552);
    assert(recorded_kernel==reinterpret_cast<const void*>(fused_lds_kernel4));
    assert(recorded.m==2048 && recorded.n==128 && recorded.k==16384 && recorded.batch==1);
    assert(recorded.split_k==16 && recorded.stride_a==16384 && recorded.stride_b==16384);
    assert(recorded.stride_c==128 && recorded.stride_sfa==2048 && recorded.stride_sfb==128);
    assert(recorded.ptr_a==reinterpret_cast<void*>(0x1000));
    assert(recorded.ptr_b==reinterpret_cast<void*>(0x2000));
    assert(recorded.ptr_sfa==reinterpret_cast<void*>(0x3000));
    assert(recorded.ptr_sfb==reinterpret_cast<void*>(0x4000));
    assert(recorded.ptr_c==reinterpret_cast<void*>(0x5000));
    assert(recorded.ptr_workspace==reinterpret_cast<float*>(24576));
    assert(recorded.ptr_counters==reinterpret_cast<int*>(28672));valid++; }
{ int before=calls;assert(call(4,16,128,128,1,
    0x1000,0x2000,0x3000,0x4000,0x5000,0,0)==0);assert(calls==before+1);
    assert(grid.x==8 && grid.y==1 && grid.z==1);
    assert(block.x==256 && block.y==1 && block.z==1 && dynamic_lds==25409);
    assert(recorded_kernel==reinterpret_cast<const void*>(fused_lds_kernel4));
    assert(recorded.m==16 && recorded.n==128 && recorded.k==128 && recorded.batch==1);
    assert(recorded.split_k==1 && recorded.stride_a==128 && recorded.stride_b==128);
    assert(recorded.stride_c==128 && recorded.stride_sfa==16 && recorded.stride_sfb==1);
    assert(recorded.ptr_a==reinterpret_cast<void*>(0x1000));
    assert(recorded.ptr_b==reinterpret_cast<void*>(0x2000));
    assert(recorded.ptr_sfa==reinterpret_cast<void*>(0x3000));
    assert(recorded.ptr_sfb==reinterpret_cast<void*>(0x4000));
    assert(recorded.ptr_c==reinterpret_cast<void*>(0x5000));
    assert(recorded.ptr_workspace==reinterpret_cast<float*>(0));
    assert(recorded.ptr_counters==reinterpret_cast<int*>(0));valid++; }
{int before=calls;assert(call(4,0,128,384,3)==1);assert(calls==before);rejected++;}
{int before=calls;assert(call(4,2049,128,384,3)==1);assert(calls==before);rejected++;}
{int before=calls;assert(call(4,16,127,384,3)==1);assert(calls==before);rejected++;}
{int before=calls;assert(call(4,16,128,129,1)==1);assert(calls==before);rejected++;}
{int before=calls;assert(call(4,16,128,384,0)==1);assert(calls==before);rejected++;}
{int before=calls;assert(call(4,16,128,384,4)==1);assert(calls==before);rejected++;}
{int before=calls;assert(call(4,16,128,384,3,0x0)==1);assert(calls==before);rejected++;}
{int before=calls;assert(call(4,16,128,384,3,0x1001)==1);assert(calls==before);rejected++;}
{int before=calls;assert(call(4,16,128,384,3,0x1000,0x2000,0x3000,0x4000,0x5000,0x0)==1);assert(calls==before);rejected++;}
{int before=calls;assert(call(4,16,128,384,3,0x1000,0x2000,0x3000,0x4000,0x5000,0x6000,0x0)==1);assert(calls==before);rejected++;}
{int before=calls;assert(call(4,16,128,384,3,0x1000,0x2000,0x3000,0x4000,0x5000,0x6001,0x7000)==1);assert(calls==before);rejected++;}
{int before=calls;assert(call(4,16,128,384,3,0x1000,0x2000,0x3000,0x4000,0x5000,0x6000,0x7001)==1);assert(calls==before);rejected++;}
{int before=calls;assert(call(4,16,128,384,3,0x1000,0x2000,0x3001)==1);assert(calls==before);rejected++;}
{int before=calls;assert(call(4,16,128,384,3,0x1000,0x2000,0x3000,0x4001)==1);assert(calls==before);rejected++;}
{ int before=calls;assert(call(5,1,128,384,3,
    0x1000,0x2000,0x3000,0x4000,0x5000,24576,28672)==0);assert(calls==before+1);
    assert(grid.x==24 && grid.y==1 && grid.z==1);
    assert(block.x==256 && block.y==1 && block.z==1 && dynamic_lds==25409);
    assert(recorded_kernel==reinterpret_cast<const void*>(fused_lds_kernel5));
    assert(recorded.m==1 && recorded.n==128 && recorded.k==384 && recorded.batch==1);
    assert(recorded.split_k==3 && recorded.stride_a==384 && recorded.stride_b==384);
    assert(recorded.stride_c==128 && recorded.stride_sfa==1 && recorded.stride_sfb==3);
    assert(recorded.ptr_a==reinterpret_cast<void*>(0x1000));
    assert(recorded.ptr_b==reinterpret_cast<void*>(0x2000));
    assert(recorded.ptr_sfa==reinterpret_cast<void*>(0x3000));
    assert(recorded.ptr_sfb==reinterpret_cast<void*>(0x4000));
    assert(recorded.ptr_c==reinterpret_cast<void*>(0x5000));
    assert(recorded.ptr_workspace==reinterpret_cast<float*>(24576));
    assert(recorded.ptr_counters==reinterpret_cast<int*>(28672));valid++; }
{ int before=calls;assert(call(5,17,256,7168,4,
    0x1000,0x2000,0x3000,0x4000,0x5000,24576,28672)==0);assert(calls==before+1);
    assert(grid.x==32 && grid.y==1 && grid.z==1);
    assert(block.x==256 && block.y==1 && block.z==1 && dynamic_lds==102286);
    assert(recorded_kernel==reinterpret_cast<const void*>(fused_lds_kernel5));
    assert(recorded.m==17 && recorded.n==256 && recorded.k==7168 && recorded.batch==1);
    assert(recorded.split_k==4 && recorded.stride_a==7168 && recorded.stride_b==7168);
    assert(recorded.stride_c==256 && recorded.stride_sfa==17 && recorded.stride_sfb==56);
    assert(recorded.ptr_a==reinterpret_cast<void*>(0x1000));
    assert(recorded.ptr_b==reinterpret_cast<void*>(0x2000));
    assert(recorded.ptr_sfa==reinterpret_cast<void*>(0x3000));
    assert(recorded.ptr_sfb==reinterpret_cast<void*>(0x4000));
    assert(recorded.ptr_c==reinterpret_cast<void*>(0x5000));
    assert(recorded.ptr_workspace==reinterpret_cast<float*>(24576));
    assert(recorded.ptr_counters==reinterpret_cast<int*>(28672));valid++; }
{ int before=calls;assert(call(5,2048,128,16384,16,
    0x1000,0x2000,0x3000,0x4000,0x5000,24576,28672)==0);assert(calls==before+1);
    assert(grid.x==512 && grid.y==1 && grid.z==1);
    assert(block.x==256 && block.y==1 && block.z==1 && dynamic_lds==101896);
    assert(recorded_kernel==reinterpret_cast<const void*>(fused_lds_kernel5));
    assert(recorded.m==2048 && recorded.n==128 && recorded.k==16384 && recorded.batch==1);
    assert(recorded.split_k==16 && recorded.stride_a==16384 && recorded.stride_b==16384);
    assert(recorded.stride_c==128 && recorded.stride_sfa==2048 && recorded.stride_sfb==128);
    assert(recorded.ptr_a==reinterpret_cast<void*>(0x1000));
    assert(recorded.ptr_b==reinterpret_cast<void*>(0x2000));
    assert(recorded.ptr_sfa==reinterpret_cast<void*>(0x3000));
    assert(recorded.ptr_sfb==reinterpret_cast<void*>(0x4000));
    assert(recorded.ptr_c==reinterpret_cast<void*>(0x5000));
    assert(recorded.ptr_workspace==reinterpret_cast<float*>(24576));
    assert(recorded.ptr_counters==reinterpret_cast<int*>(28672));valid++; }
{ int before=calls;assert(call(5,16,128,128,1,
    0x1000,0x2000,0x3000,0x4000,0x5000,0,0)==0);assert(calls==before+1);
    assert(grid.x==8 && grid.y==1 && grid.z==1);
    assert(block.x==256 && block.y==1 && block.z==1 && dynamic_lds==25409);
    assert(recorded_kernel==reinterpret_cast<const void*>(fused_lds_kernel5));
    assert(recorded.m==16 && recorded.n==128 && recorded.k==128 && recorded.batch==1);
    assert(recorded.split_k==1 && recorded.stride_a==128 && recorded.stride_b==128);
    assert(recorded.stride_c==128 && recorded.stride_sfa==16 && recorded.stride_sfb==1);
    assert(recorded.ptr_a==reinterpret_cast<void*>(0x1000));
    assert(recorded.ptr_b==reinterpret_cast<void*>(0x2000));
    assert(recorded.ptr_sfa==reinterpret_cast<void*>(0x3000));
    assert(recorded.ptr_sfb==reinterpret_cast<void*>(0x4000));
    assert(recorded.ptr_c==reinterpret_cast<void*>(0x5000));
    assert(recorded.ptr_workspace==reinterpret_cast<float*>(0));
    assert(recorded.ptr_counters==reinterpret_cast<int*>(0));valid++; }
{int before=calls;assert(call(5,0,128,384,3)==1);assert(calls==before);rejected++;}
{int before=calls;assert(call(5,2049,128,384,3)==1);assert(calls==before);rejected++;}
{int before=calls;assert(call(5,16,127,384,3)==1);assert(calls==before);rejected++;}
{int before=calls;assert(call(5,16,128,129,1)==1);assert(calls==before);rejected++;}
{int before=calls;assert(call(5,16,128,384,0)==1);assert(calls==before);rejected++;}
{int before=calls;assert(call(5,16,128,384,4)==1);assert(calls==before);rejected++;}
{int before=calls;assert(call(5,16,128,384,3,0x0)==1);assert(calls==before);rejected++;}
{int before=calls;assert(call(5,16,128,384,3,0x1001)==1);assert(calls==before);rejected++;}
{int before=calls;assert(call(5,16,128,384,3,0x1000,0x2000,0x3000,0x4000,0x5000,0x0)==1);assert(calls==before);rejected++;}
{int before=calls;assert(call(5,16,128,384,3,0x1000,0x2000,0x3000,0x4000,0x5000,0x6000,0x0)==1);assert(calls==before);rejected++;}
{int before=calls;assert(call(5,16,128,384,3,0x1000,0x2000,0x3000,0x4000,0x5000,0x6001,0x7000)==1);assert(calls==before);rejected++;}
{int before=calls;assert(call(5,16,128,384,3,0x1000,0x2000,0x3000,0x4000,0x5000,0x6000,0x7001)==1);assert(calls==before);rejected++;}
{int before=calls;assert(call(5,16,128,384,3,0x1000,0x2000,0x3001)==1);assert(calls==before);rejected++;}
{int before=calls;assert(call(5,16,128,384,3,0x1000,0x2000,0x3000,0x4001)==1);assert(calls==before);rejected++;}
assert(call(6,16,128,384,3)==1);rejected++;
std::printf("%d %d\n",valid,rejected);return 0;}
