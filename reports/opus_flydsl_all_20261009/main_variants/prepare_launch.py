#!/usr/bin/env python3
"""Emit a finite private ABI; independent shapes use the same current source baseline."""
from pathlib import Path
import json
H=Path(__file__).resolve().parent
variants=[]
def add(id,name,kernel,trait,bm,bn=128,k=0,pin=False,parents=None):
 variants.append(dict(id=id,name=name,kernel=kernel,traits=trait,tile_M=bm,tile_N=bn,fixed_K=k,pin=pin,parents=parents or [9020,9021,9022,9024]))
for id,bm in [(92000,64),(92001,96),(92002,128)]:
 add(id,f'bm{bm}_s2_panel32_runtime','opus_private_main_geometry_kernel',f'opus_private_geometry_traits<{bm},2,32>',bm)
for id,bm,k,panel in [(92003,96,384,8),(92004,96,768,8),(92005,128,1536,16),(92006,160,1536,16),(92007,96,3072,32),(92008,96,7168,32),(92009,128,7168,32),(92010,128,3072,32)]:
 add(id,f'bm{bm}_s2_panel{panel}_fixed{k}','opus_private_main_geometry_kernel',f'opus_private_geometry_traits<{bm},2,{panel},{k}>',bm,k=k)
add(92011,'9024_fixed7168_groupM8','gemm_a8w8_mxfp8_scale_4wave_64x64_kernel','opus_gemm_mxscale_bpreshuffle_4wave_64x64_traits_base_gfx950<64,7168,8>',64,64,7168,parents=[9024])
add(92012,'9024_fixed7168_groupM16','gemm_a8w8_mxfp8_scale_4wave_64x64_kernel','opus_gemm_mxscale_bpreshuffle_4wave_64x64_traits_base_gfx950<64,7168,16>',64,64,7168,parents=[9024])
# Frozen baselines are selected exactly by the published current path within this pool.
add(9022,'baseline9022','gemm_a8w8_mxfp8_scale_4wave_160x128_kernel','opus_gemm_mxscale_bpreshuffle_4wave_160x128_traits_gfx950',160,parents=[9022])
add(9021,'baseline9021','gemm_a8w8_mxfp8_scale_4wave_128x128_kernel','opus_gemm_mxscale_bpreshuffle_4wave_128x128_traits_gfx950',128,parents=[9021])
add(9020,'baseline9020_runtime','gemm_a8w8_mxfp8_scale_8wave_192x256_kernel','opus_gemm_mxscale_bpreshuffle_8wave_traits_gfx950<>',192,256,parents=[9020])
add(91920,'baseline9020_narrow','gemm_a8w8_mxfp8_scale_8wave_192x256_kernel','opus_gemm_mxscale_bpreshuffle_8wave_traits_gfx950<128,128,64>',128,128,parents=[9020])
add(9024,'baseline9024','gemm_a8w8_mxfp8_scale_4wave_64x64_kernel','opus_gemm_mxscale_bpreshuffle_4wave_64x64_traits_base_gfx950<64,7168,4>',64,64,7168,parents=[9024])
for parent in (9000,9001):
 for j,(k,panel) in enumerate([(384,16),(1536,16),(3072,32),(7168,64),(16384,64)]):
  add(92100+(parent-9000)*10+j,f'{parent}_fixed{k}_panel{panel}','opus_private_pin::gemm_a8w8_mxfp8_scale_kernel_private_fixed',f'opus_private_pin_fixed_traits<{k},{panel},{str(parent==9001).lower()}>',256,256,k,True,[parent])
add(92120,'9011_fixed7168','opus_private_pin::gemm_a8w8_mxfp8_scale_4wave_256x256_padded_m_kernel_private_fixed','opus_private_pad_pin_fixed_traits<7168>',256,256,7168,True,[9011])
add(9000,'baseline9000','gemm_a8w8_mxfp8_scale_kernel','opus_gemm_mxscale_bpreshuffle_4wave_traits_gfx950',256,256,pin=True,parents=[9000])
add(9001,'baseline9001','gemm_a8w8_mxfp8_scale_kernel','opus_gemm_mxscale_bpreshuffle_4wave_traits_scale_reset_gfx950',256,256,pin=True,parents=[9001])
add(9011,'baseline9011','gemm_a8w8_mxfp8_scale_4wave_256x256_padded_m_kernel','opus_gemm_mxscale_bpreshuffle_4wave_256x256_padded_m_traits_unroll4_gfx950',256,256,pin=True,parents=[9011])
common='''#include <hip/hip_runtime.h>
#include <cstdint>
#define __HIPCC_RTC__ 1
#include "contract.h"
#include "candidate/traits.cuh"
'''
for pin in (False,True):
 items=[v for v in variants if v['pin']==pin]
 headers=['candidate/pin.cuh','candidate/pad_pin.cuh','opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_4wave_256x256_padded_m_gfx950.cuh'] if pin else ['candidate/geometry.cuh','opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_4wave_160x128_gfx950.cuh','opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_4wave_64x64_gfx950.cuh']
 src=common+''.join(f'#include "{h}"\n' for h in headers)
 src+='''extern "C" __attribute__((visibility("default")))
int launch(int kid,const void* a,const void* b,const void* sfa,const void* sfb,void* c,int m,int n,int k,void* stream) {
 if(!main_pointers_valid(a,b,sfa,sfb,c) || !main_shape_valid(m,n,k,128,0))return int(hipErrorInvalidValue);
 opus_gemm_mxscale_bpreshuffle_kargs_gfx950 args{};
 args.ptr_a=a;args.ptr_b=b;args.ptr_sfa=sfa;args.ptr_sfb=sfb;args.ptr_c=c;
 args.m=m;args.n=n;args.k=k;args.batch=1;
 args.stride_a=k;args.stride_b=k;args.stride_c=n;args.stride_sfa=m;args.stride_sfb=k/128;
 args.stride_a_batch=int64_t(m)*k;args.stride_b_batch=int64_t(n)*k;args.stride_c_batch=int64_t(m)*n;
 args.stride_sfa_batch=int64_t(m)*(k/128);args.stride_sfb_batch=int64_t(n/128)*(k/128);
 // Extent guard occurs before these values are consumed by a kernel.
 const auto s=reinterpret_cast<hipStream_t>(stream);
 switch(kid) {
'''
 for v in items:
  src+=f''' case {v['id']}:{{
 if(!main_shape_valid(m,n,k,{v['tile_N']},{v['fixed_K']})){ '{' }return int(hipErrorInvalidValue);{ '}' }
'''
  if pin and v['id']!=9011 and v['id']!=92120: src+=' if(m%256)return int(hipErrorInvalidValue);\n'
  src+=f''' using T={v['traits']};
 {v['kernel']}<T><<<dim3(n/T::B_N,1+(m-1)/T::B_M),T::BLOCK_SIZE,0,s>>>(args);
 break;}}
'''
 src+=' default:return int(hipErrorInvalidValue);\n }return int(hipGetLastError());\n}\n'
 (H/('launch_pin.hip' if pin else 'launch.hip')).write_text(src)
(H/'variants.json').write_text(json.dumps(variants,indent=2)+'\n')
print('Emitted',sum(v['id']>=92000 for v in variants),'new candidates and',sum(v['id']<92000 for v in variants),'controls')
