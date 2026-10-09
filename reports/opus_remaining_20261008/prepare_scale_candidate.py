#!/usr/bin/env python3
"""Isolated 9020/9022 raw scale request overlap, preserving matrix schedule."""
from pathlib import Path
import difflib
import hashlib
import json
import re
import shutil

OUT=Path(__file__).resolve().parent/'scale_issue'
ROOT=OUT.parents[2]
SOURCE=ROOT/'csrc/opus_gemm/include'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def modify(text, parent):
    begin=text.index('    // Scale A global memory -> VGPR -> LDS, retaining raw E8M0 bytes.')
    end=text.index('    // Scale LDS -> VGPR;',begin)
    old=text[begin:end]
    a=old[:old.index('    // Scale B global memory')]
    b=old[old.index('    // Scale B global memory'):]
    a=a.replace('auto load_sfa_panel =','auto issue_sfa_panel =').replace(
        'vector_t<D_SF, T::VEC_SCALE_A> raw;','auto& raw = raw_sfa_panel[pass];')
    a=a.replace('                store<T::VEC_SCALE_A>(s_sfa, raw, smem_offset);\n','')
    if parent==9020:
        b=b.replace('auto load_sfb_panel =','auto issue_sfb_panel =')
        b=b.replace('const unsigned lo =','raw_sfb_panel[0] =').replace('[0];\n            unsigned hi = 0;', '[0];\n            raw_sfb_panel[1] = 0;')
        b=b.replace('hi = load<1>', 'raw_sfb_panel[1] = load<1>')
        b=b[:b.index('            // One word per K128 group:')]+'''        }
    };
    auto publish_sfa_panel = [&](int panel_k_begin) {
        const int local_k_group = wave_id * T::SFA_K_COLUMNS_PER_WAVE + lane_id / T::SFA_THREADS_PER_GROUP;
        const int k_group = panel_k_begin + local_k_group;
        static_for<T::SFA_PASSES>([&](auto pass_i) {
            constexpr int pass = decltype(pass_i)::value;
            const int local_row = sfa_smem_offsets[pass] - local_k_group * T::B_M;
            const int smem_offset = sfa_smem_offsets[pass] + panel_k_begin * T::B_M;
            if (local_row < T::B_M && smem_offset < T::SFA_BYTES && k_group < scale_k_groups)
                store<T::VEC_SCALE_A>(s_sfa, raw_sfa_panel[pass], smem_offset);
        });
    };
    auto publish_sfb_panel = [&](int panel_k_begin) {
        const int k_group = panel_k_begin + wave_id * T::WARP_SIZE + lane_id;
        if (k_group < scale_k_groups) {
            const vector_t<D_SF_PACK, 1> packed{unsigned(raw_sfb_panel[0]) | (unsigned(raw_sfb_panel[1]) << 8)};
            store<1>(s_sfb, packed, sfb_smem_offsets[0]);
        }
    };
'''
        raw_decl='    vector_t<D_SF, 2> raw_sfb_panel;\n'
    else:
        b='''    auto issue_sfb_panel = [&](int panel_k_begin) {
        const int local_k_group = sfb_smem_offsets[0];
        if (local_k_group < T::SCALE_PANEL && panel_k_begin + local_k_group < loops)
            raw_sfb_panel = load<1>(g_sfb, sfb_gmem_offsets[0] + gsfb_offset(panel_k_begin));
    };
    auto publish_sfa_panel = [&](int panel_k_begin) {
        static_for<T::SFA_PASSES>([&](auto pass_i) {
            constexpr int pass = decltype(pass_i)::value;
            const int smem_offset = opus::get<pass>(sfa_smem_offsets)[0];
            const int local_k_group = smem_offset / T::B_M;
            if (smem_offset < T::SFA_BYTES && panel_k_begin + local_k_group < loops)
                store<T::VEC_SCALE_A>(s_sfa, raw_sfa_panel[pass], smem_offset);
        });
    };
    auto publish_sfb_panel = [&](int panel_k_begin) {
        const int local_k_group = sfb_smem_offsets[0];
        if (local_k_group < T::SCALE_PANEL && panel_k_begin + local_k_group < loops)
            store<1>(s_sfb, raw_sfb_panel, sfb_smem_offsets[0]);
    };
'''
        raw_decl='    vector_t<D_SF, 1> raw_sfb_panel;\n'
    if parent == 9020:
        a=old[:old.index('    // Scale B global memory')]
        b=b[:b.index('    auto publish_sfa_panel')]+b[b.index('    auto publish_sfb_panel'):]
        new=raw_decl+a+b
    else:
        new='    array<vector_t<D_SF, T::VEC_SCALE_A>, T::SFA_PASSES> raw_sfa_panel;\n'+raw_decl+a+b
    candidate=text[:begin]+new+text[end:]
    if parent == 9020:
        block_begin=candidate.index('    // Spread short-K scale loads')
        block_end=candidate.index('    __builtin_amdgcn_sched_barrier(0);',block_begin)
        old_block=candidate[block_begin:block_end]
        replacement='    issue_sfb_panel(0);\n    __builtin_amdgcn_sched_barrier(0);\n'+old_block.replace('    load_sfb_panel(0);\n','    publish_sfb_panel(0);\n')
        candidate=candidate[:block_begin]+replacement+candidate[block_end:]
    for arg,indent in ([] if parent == 9020 else [('0','    '),('tile_k + 1','            ')]):
        original=indent+f'load_sfa_panel({arg});\n'+indent+f'load_sfb_panel({arg});\n'
        if original not in candidate:
            if parent==9020 and arg!='0': continue
            raise RuntimeError(f'Missing exact call block {parent} {arg}')
        replacement=''.join(indent+line+'\n' for line in [
            f'issue_sfa_panel({arg});',f'issue_sfb_panel({arg});',
            '__builtin_amdgcn_sched_barrier(0);',f'publish_sfa_panel({arg});',f'publish_sfb_panel({arg});'])
        candidate=candidate.replace(original,replacement,1)
    sync=r'(?:s_waitcnt_vmcnt|s_waitcnt_lgkmcnt|__builtin_amdgcn_s_barrier)\([^;]*?\);'
    assert re.findall(sync,text)==re.findall(sync,candidate)
    assert text.split('    // Epilogue',1)[1]==candidate.split('    // Epilogue',1)[1]
    return candidate


def main():
    if OUT.exists(): raise RuntimeError('Refuse to overwrite existing isolated experiment')
    OUT.mkdir(parents=True)
    for side in ['baseline','candidate']:
        shutil.copytree(SOURCE,OUT/side/'include')
    rows=[]
    for parent,stem in [(9020,'8wave_192x256'),(9022,'4wave_160x128')]:
        relative=Path(f'include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_{stem}_gfx950.cuh')
        path=OUT/'baseline'/relative
        text=path.read_text();candidate=modify(text,parent)
        (OUT/'candidate'/relative).write_text(candidate)
        (OUT/f'kid{parent}.diff').write_text(''.join(difflib.unified_diff(
            text.splitlines(True),candidate.splitlines(True),fromfile='baseline/'+str(relative),tofile='candidate/'+str(relative))))
        rows.append({'parent':parent,'path':str(relative),'baseline_sha256':sha(text.encode()),'candidate_sha256':sha(candidate.encode())})
    launcher='''#include <hip/hip_runtime.h>
#include <cstdint>
#define __HIPCC_RTC__ 1
#include "include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_8wave_192x256_gfx950.cuh"
#include "include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_4wave_160x128_gfx950.cuh"
template<class T> void launch_9020(opus_gemm_mxscale_bpreshuffle_kargs_gfx950 args, hipStream_t stream) {
 gemm_a8w8_mxfp8_scale_8wave_192x256_kernel<T><<<dim3(args.n/T::B_N,(args.m+T::B_M-1)/T::B_M),T::BLOCK_SIZE,0,stream>>>(args);
}
extern "C" __attribute__((visibility("default")))
int launch(int kid,const void* a,const void* b,const void* sfa,const void* sfb,void* c,int m,int n,int k,void* stream) {
 constexpr int64_t limit=INT32_MAX;
 if ((kid!=9020&&kid!=9022)||m<=0||n<=0||k<=0||k>16384||k%128||m%16||
     n%(kid==9020?256:128)||int64_t(m)*k>limit||int64_t(n)*k>limit||int64_t(m)*n*2>limit||
     !a||!b||!sfa||!sfb||!c||reinterpret_cast<uintptr_t>(a)%16||reinterpret_cast<uintptr_t>(b)%16||
     reinterpret_cast<uintptr_t>(sfa)%16||reinterpret_cast<uintptr_t>(c)%16) return int(hipErrorInvalidValue);
 opus_gemm_mxscale_bpreshuffle_kargs_gfx950 args{};
 args.ptr_a=a;args.ptr_b=b;args.ptr_sfa=sfa;args.ptr_sfb=sfb;args.ptr_c=c;
 args.m=m;args.n=n;args.k=k;args.batch=1;args.stride_a=k;args.stride_b=k;args.stride_c=n;
 args.stride_sfa=m;args.stride_sfb=k/128;args.stride_a_batch=m*k;args.stride_b_batch=n*k;
 args.stride_c_batch=m*n;args.stride_sfa_batch=m*(k/128);args.stride_sfb_batch=(n/128)*(k/128);
 auto hip_stream=reinterpret_cast<hipStream_t>(stream);
 if(kid==9022) {using T=opus_gemm_mxscale_bpreshuffle_4wave_160x128_traits_gfx950;
 gemm_a8w8_mxfp8_scale_4wave_160x128_kernel<T><<<dim3(n/128,(m+159)/160),256,0,hip_stream>>>(args);}
 else if(m>=8192&&n<=1024&&k<=8192&&(((m+127)/128)*(n/128)<=512||((m+191)/192)*(n/256)>256))
 launch_9020<opus_gemm_mxscale_bpreshuffle_8wave_traits_gfx950<128,128,64>>(args,hip_stream);
 else if(m>=1024&&k==384) launch_9020<opus_gemm_mxscale_bpreshuffle_8wave_traits_gfx950<192,256,8,384>>(args,hip_stream);
 else if(m>=1024&&k==768) launch_9020<opus_gemm_mxscale_bpreshuffle_8wave_traits_gfx950<192,256,8,768>>(args,hip_stream);
 else if(m>=1024&&k==1536) launch_9020<opus_gemm_mxscale_bpreshuffle_8wave_traits_gfx950<192,256,32,1536>>(args,hip_stream);
 else if(m>=1024&&k==3072) launch_9020<opus_gemm_mxscale_bpreshuffle_8wave_traits_gfx950<192,256,32,3072>>(args,hip_stream);
 else if(m>=1024&&k==7168) launch_9020<opus_gemm_mxscale_bpreshuffle_8wave_traits_gfx950<192,256,64,7168>>(args,hip_stream);
 else launch_9020<opus_gemm_mxscale_bpreshuffle_8wave_192x256_traits_gfx950>(args,hip_stream);
 return int(hipGetLastError());
}
'''
    for side in ['baseline','candidate']:(OUT/side/'launch.hip').write_text(launcher)
    source_rows=[]
    for path in sorted((OUT/'baseline/include').rglob('*')):
        if path.is_file():
            rel=path.relative_to(OUT/'baseline');other=OUT/'candidate'/rel
            source_rows.append({'path':str(rel),'baseline_sha256':sha(path.read_bytes()),'candidate_sha256':sha(other.read_bytes())})
    manifest={'status':'isolated_prepared_not_built','baseline':'Oct8 current selected',
              'parents':[9020,9022],'production_modified':False,'changes':rows,
              'source_files':source_rows,'script_sha256':sha(Path(__file__).read_bytes()),
              'scope':'raw SFA/SFB issue before dependent LDS publish; matrix schedule, scale bytes/layout/tail, all runtime synchronization preserved'}
    (OUT/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'status':manifest['status'],'out':str(OUT),'source_files':len(source_rows)}))


if __name__=='__main__':main()
