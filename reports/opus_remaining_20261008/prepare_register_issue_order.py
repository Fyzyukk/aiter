#!/usr/bin/env python3
"""Prepare exact runtime9051 scale/matrix issue-order experiment."""
import difflib
import hashlib
import json
from pathlib import Path
import re
import shutil

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=HERE/'register_issue_order'
SOURCE=ROOT/'csrc/opus_gemm/include'
FORMAL=ROOT/'reports/opus_resume_20261008/formal_selected/identity_audit.json'


def sha(data):return hashlib.sha256(data).hexdigest()


def main():
    if OUT.exists():raise RuntimeError('Refuse to overwrite existing experiment')
    OUT.mkdir()
    for side in ['baseline','candidate']:shutil.copytree(SOURCE,OUT/side/'include')
    rel=Path('include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_small_register_gfx950.cuh')
    baseline=(OUT/'baseline'/rel).read_text()
    begin=baseline.index('        static_for<T::E_M>([&](auto mi) {',baseline.index('    auto prefetch ='))
    end=baseline.index('\n    };\n    auto compute',begin)
    old=baseline[begin:end]
    selected='''        if constexpr (std::is_same_v<T, opus_gemm_small_register_traits_gfx950<
                16, 32, 1, 1, 3, 4, 3, 3, 0, false, false>>) {
            // Issue the same raw scales first; no operand is consumed until
            // the original compute queues/waits and K-wave reduction below.
            static_for<T::E_M>([&](auto mi) {
                constexpr int m = decltype(mi)::value;
                const int r = (m * T::T_M + wm) * T::W_M + lane % T::W_M;
                const int sf_offset = r + row < args.m ? (kt * T::W_K / 128) * args.stride_sfa + row + r : -1;
                q.sfa[m] = load<1>(gsa, sf_offset)[0];
            });
            static_for<T::E_N>([&](auto ni) {
                constexpr int n = decltype(ni)::value;
                const int nr = (n * T::T_N + wn) * T::W_N;
                q.sfb[n] = load<1>(gsb, ((col + nr) / 128) * args.stride_sfb + kt * T::W_K / 128)[0];
            });
            __builtin_amdgcn_sched_barrier(0);
            static_for<T::E_M>([&](auto mi) {
                constexpr int m = decltype(mi)::value;
                const int r = (m * T::T_M + wm) * T::W_M + lane % T::W_M;
                const int offset = r * args.stride_a + (lane / T::W_M) * 16;
                set_slice(q.a[m], load<16>(ga, offset, kt * T::W_K), 0_I, 16_I);
                set_slice(q.a[m], load<16>(ga, offset + T::W_K / 2, kt * T::W_K), 16_I, 32_I);
            });
            static_for<T::E_N>([&](auto ni) {
                constexpr int n = decltype(ni)::value;
                const int nr = (n * T::T_N + wn) * T::W_N;
                const int offset = nr * args.stride_b + lane * 16;
                set_slice(q.b[n], load<16>(gb, offset, kt * T::W_K * 16, number<T::B_CACHE>{}), 0_I, 16_I);
                set_slice(q.b[n], load<16>(gb, offset + T::W_K * 8, kt * T::W_K * 16, number<T::B_CACHE>{}), 16_I, 32_I);
            });
        } else {
'''
    new=selected+'\n'.join('    '+line for line in old.splitlines())+'\n        }'
    candidate=baseline[:begin]+new+baseline[end:]
    sync=r'(?:s_waitcnt_vmcnt|s_waitcnt_lgkmcnt|__builtin_amdgcn_s_barrier)\([^;]*?\);'
    assert re.findall(sync,baseline)==re.findall(sync,candidate)
    assert baseline.split('    auto compute =',1)[1]==candidate.split('    auto compute =',1)[1]
    (OUT/'candidate'/rel).write_text(candidate)
    (OUT/'candidate.diff').write_text(''.join(difflib.unified_diff(baseline.splitlines(True),candidate.splitlines(True),fromfile='baseline/'+str(rel),tofile='candidate/'+str(rel))))
    source_rows=[]
    for path in sorted((OUT/'baseline/include').rglob('*')):
        if path.is_file():
            relative=path.relative_to(OUT/'baseline')
            source_rows.append({'path':str(relative),'baseline_sha256':sha(path.read_bytes()),'candidate_sha256':sha((OUT/'candidate'/relative).read_bytes())})
    changed=[r for r in source_rows if r['baseline_sha256']!=r['candidate_sha256']]
    assert [r['path'] for r in changed]==[str(rel)]
    formal=json.loads(FORMAL.read_text());tus=[]
    for p in formal['parents']:
        if p['parent_id']<9040:continue
        obj=Path(p['candidate_object']);staging=obj.parent.parent/'blob.staging';source=staging/'instances'/obj.name.replace('.cuda.o','.cu')
        tus.append({'parent_id':p['parent_id'],'official_object':str(obj),'source':str(source),'source_sha256':sha(source.read_bytes()),'staging':str(staging),'variants':[v['candidate'] for v in p['variants']]})
    manifest={'status':'isolated_prepared_not_built','production_modified':False,'mechanism':'Runtime9051 same tile SFA/SFB byte requests before A/B matrix requests',
              'selected_parent_ids':[9051],'selected_symbols':[v['name'] for t in tus if t['parent_id']==9051 for v in t['variants']],
              'device_tus':tus,'source_files':source_rows,'changes':changed,'formal_identity':{'path':str(FORMAL),'sha256':sha(FORMAL.read_bytes())},
              'runtime_wait_barrier_sequence_unchanged':True,'compute_queue_retirement_reduction_output_unchanged':True,
              'scope':'Only exactruntime9051<Q3,waveK4,OUTPUT3,K0,ReuseBScale=false> type. Same byte requests, matrix slices, scale/tail guards, cache policy, queue lifecycle and reduction/output. Compiler sched_barrier only locks order, no runtime wait/barrier added.',
              'script_sha256':sha(Path(__file__).read_bytes())}
    (OUT/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    build=(HERE/'ring_overlap/build_and_audit.py').read_text()
    build=build.replace("t['parent_id'] not in [9046, 9055]","t['parent_id'] != 9051")
    build=build.replace("assert sum(c['selected'] for c in checks) == 2","assert sum(c['selected'] for c in checks) == 1")
    build=build.replace("'unselected_device_entries_unchanged': len(checks) - 2","'unselected_device_entries_unchanged': len(checks) - 1")
    build=build.replace("'unselected_unchanged': len(checks)-2","'unselected_unchanged': len(checks)-1")
    host='''#define __HIPCC_RTC__ 1
#include <opus/hip_minimal.hpp>
#include <cstdint>
#include "gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_small_gfx950.cuh"
template<class T> __global__ void gemm_a8w8_mxfp8_scale_small_register_kernel(opus_gemm_mxscale_bpreshuffle_kargs_gfx950);
using R9051=opus_gemm_small_register_traits_gfx950<16,32,1,1,3,4,3,3,0,false,false>;
#if !defined(__HIP_DEVICE_COMPILE__)
extern "C" __attribute__((visibility("default"))) int launch(int kid,const void* a,const void* b,const void* sfa,const void* sfb,void* c,int m,int n,int k,void* stream){
 constexpr int64_t limit=INT32_MAX;
 if(kid!=9051||!a||!b||!sfa||!sfb||!c||m<1||m>512||n<128||n%128||k<128||k>16384||k%128||int64_t(m)*k>limit||int64_t(n)*k>limit||int64_t(m)*n*2>limit||reinterpret_cast<uintptr_t>(a)%16||reinterpret_cast<uintptr_t>(b)%16||reinterpret_cast<uintptr_t>(sfa)%16||reinterpret_cast<uintptr_t>(c)%16)return 1;
 opus_gemm_mxscale_bpreshuffle_kargs_gfx950 args{};
 args.ptr_a=a;args.ptr_b=b;args.ptr_sfa=sfa;args.ptr_sfb=sfb;args.ptr_c=c;args.m=m;args.n=n;args.k=k;args.batch=1;
 args.stride_a=k;args.stride_b=k;args.stride_c=n;args.stride_sfa=m;args.stride_sfb=k/128;
 args.stride_a_batch=m*k;args.stride_b_batch=n*k;args.stride_c_batch=m*n;args.stride_sfa_batch=m*(k/128);args.stride_sfb_batch=(n/128)*(k/128);
 gemm_a8w8_mxfp8_scale_small_register_kernel<R9051><<<dim3(n/32,(m+15)/16),dim3(256),0,reinterpret_cast<hipStream_t>(stream)>>>(args);
 return hipGetLastError();
}
#endif
'''
    start=build.index('def host_source():');end=build.index('\n\ndef main():',start)
    build=build[:start]+'def host_source():\n    return '+repr(host)+'\n'+build[end:]
    (OUT/'build_and_audit.py').write_text(build)
    print(json.dumps({'status':manifest['status'],'out':str(OUT),'selected_symbols':len(manifest['selected_symbols'])}))


if __name__=='__main__':main()
