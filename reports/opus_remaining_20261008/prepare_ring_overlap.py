#!/usr/bin/env python3
"""Prepare exact9046/9055 consumer LDS/future matrix issue-order experiment."""
import difflib
import hashlib
import json
from pathlib import Path
import re
import shutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / 'ring_overlap'
SOURCE = ROOT / 'csrc/opus_gemm/include'
FORMAL = ROOT / 'reports/opus_resume_20261008/formal_selected/identity_audit.json'


def sha(data): return hashlib.sha256(data).hexdigest()


def main():
    if OUT.exists(): raise RuntimeError('Refuse to overwrite existing experiment')
    OUT.mkdir()
    for side in ['baseline', 'candidate']: shutil.copytree(SOURCE, OUT / side / 'include')
    rel = Path('include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_small_lds_gfx950.cuh')
    baseline = (OUT / 'baseline' / rel).read_text()
    candidate = baseline.replace('    auto step = [&](auto si, int kt, auto ring) {', '''    // Only these existing non-register scale ring types move their unchanged
    // future matrix issue before current operand reads, below group visibility.
    constexpr bool selected_ring_prefetch_before_read =
        std::is_same_v<T, opus_gemm_small_lds_traits_gfx950<64, 128, 4, 2, 6, 2, 2,
            false, false, false, false, false, 1, 4, 128, 0, 0, false>> ||
        std::is_same_v<T, opus_gemm_small_lds_traits_gfx950<32, 64, 1, 4, 12, 4, 2,
            false, false, true, false, false, 1, 4, 128, 0, 0, false>>;
    constexpr bool prefetch_before_read = T::PREFETCH_BEFORE_READ || selected_ring_prefetch_before_read;
    auto step = [&](auto si, int kt, auto ring) {''', 1)
    candidate = candidate.replace('decltype(ring)::value && T::PREFETCH_BEFORE_READ', 'decltype(ring)::value && prefetch_before_read')
    candidate = candidate.replace('decltype(ring)::value && !T::PREFETCH_BEFORE_READ', 'decltype(ring)::value && !prefetch_before_read')
    assert candidate != baseline
    sync = r'(?:s_waitcnt_vmcnt|s_waitcnt_lgkmcnt|__builtin_amdgcn_s_barrier)\([^;]*?\);'
    assert re.findall(sync, baseline) == re.findall(sync, candidate)
    assert baseline.split('    auto step =',1)[0] in candidate
    assert baseline.split('    if (loops <= T::NUM_STAGES) {',1)[1] == candidate.split('    if (loops <= T::NUM_STAGES) {',1)[1]
    (OUT / 'candidate' / rel).write_text(candidate)
    (OUT / 'candidate.diff').write_text(''.join(difflib.unified_diff(baseline.splitlines(True), candidate.splitlines(True),
                        fromfile='baseline/' + str(rel), tofile='candidate/' + str(rel))))
    source_rows = []
    for path in sorted((OUT / 'baseline/include').rglob('*')):
        if path.is_file():
            relative = path.relative_to(OUT / 'baseline')
            source_rows.append({'path': str(relative), 'baseline_sha256': sha(path.read_bytes()),
                                'candidate_sha256': sha((OUT / 'candidate' / relative).read_bytes())})
    changed = [r for r in source_rows if r['baseline_sha256'] != r['candidate_sha256']]
    assert [r['path'] for r in changed] == [str(rel)]
    formal = json.loads(FORMAL.read_text())
    tus = []
    for p in formal['parents']:
        if p['parent_id'] < 9040: continue
        obj = Path(p['candidate_object']); staging = obj.parent.parent / 'blob.staging'
        source = staging / 'instances' / obj.name.replace('.cuda.o','.cu')
        tus.append({'parent_id': p['parent_id'], 'official_object': str(obj), 'source': str(source),
                    'source_sha256': sha(source.read_bytes()), 'staging': str(staging),
                    'variants': [v['candidate'] for v in p['variants']]})
    manifest = {'status': 'isolated_prepared_not_built', 'production_modified': False,
                'mechanism': 'Exact9046/9055 ring step future matrix issue before current LDS operand reads',
                'selected_parent_ids': [9046, 9055],
                'selected_symbols': [v['name'] for t in tus if t['parent_id'] in [9046,9055] for v in t['variants']],
                'device_tus': tus, 'source_files': source_rows, 'changes': changed,
                'formal_identity': {'path': str(FORMAL), 'sha256': sha(FORMAL.read_bytes())},
                'runtime_wait_barrier_sequence_unchanged': True, 'initial_prologue_and_drain_output_unchanged': True,
                'scope': 'Only explicit current9046/S6C2 and9055/S12C4 types; same request calls/slot indices/guards. Future tile remains in retired ring slots and below unchanged group barrier. Shared helpers, current scale loads, MFMA order, output and all other traits unchanged.',
                'script_sha256': sha(Path(__file__).read_bytes())}
    (OUT / 'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    build = (HERE / 'fine_startup/build_and_audit.py').read_text()
    build = build.replace("t['parent_id'] != 9062", "t['parent_id'] not in [9046, 9055]")
    build = build.replace("selected_name = next(v['name'] for t in tuples for v in t['variants'] if source['selected_trait'] in v['demangled'])", "selected_names = set(source['selected_symbols'])")
    build = build.replace('selected = name == selected_name', 'selected = name in selected_names')
    build = build.replace("assert sum(c['selected'] for c in checks) == 1", "assert sum(c['selected'] for c in checks) == 2")
    build = build.replace('assert selected_name in {r[\'name\'] for r in linked}', "assert selected_names <= {r['name'] for r in linked}")
    build = build.replace("'selected_symbol': selected_name", "'selected_symbols': sorted(selected_names)")
    build = build.replace("'unselected_device_entries_unchanged': len(checks) - 1", "'unselected_device_entries_unchanged': len(checks) - 2")
    build = build.replace("'unselected_unchanged': len(checks)-1", "'unselected_unchanged': len(checks)-2")
    start=build.index('def host_source():');end=build.index('\n\ndef main():',start)
    host = '''#define __HIPCC_RTC__ 1
#include <opus/hip_minimal.hpp>
#include <cstdint>
#include "gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_small_gfx950.cuh"
template<class T> __global__ void gemm_a8w8_mxfp8_scale_small_lds_kernel(opus_gemm_mxscale_bpreshuffle_kargs_gfx950);
using L9046 = opus_gemm_small_lds_traits_gfx950<64,128,4,2,6,2,2,false,false,false,false,false,1,4,128,0,0,false>;
using L9055 = opus_gemm_small_lds_traits_gfx950<32,64,1,4,12,4,2,false,false,true,false,false,1,4,128,0,0,false>;
#if !defined(__HIP_DEVICE_COMPILE__)
template<class T> int invoke(const void* a,const void* b,const void* sfa,const void* sfb,void* c,int m,int n,int k,void* stream){
 constexpr int64_t limit=INT32_MAX;
 if(!a||!b||!sfa||!sfb||!c||m<1||m>2048||n<T::B_N||n%T::B_N||k<128||k>16384||k%128||int64_t(m)*k>limit||int64_t(n)*k>limit||int64_t(m)*n*2>limit||reinterpret_cast<uintptr_t>(a)%16||reinterpret_cast<uintptr_t>(b)%16||reinterpret_cast<uintptr_t>(sfa)%16||reinterpret_cast<uintptr_t>(c)%16)return 1;
 opus_gemm_mxscale_bpreshuffle_kargs_gfx950 args{};
 args.ptr_a=a;args.ptr_b=b;args.ptr_sfa=sfa;args.ptr_sfb=sfb;args.ptr_c=c;args.m=m;args.n=n;args.k=k;args.batch=1;
 args.stride_a=k;args.stride_b=k;args.stride_c=n;args.stride_sfa=m;args.stride_sfb=k/128;
 args.stride_a_batch=m*k;args.stride_b_batch=n*k;args.stride_c_batch=m*n;args.stride_sfa_batch=m*(k/128);args.stride_sfb_batch=(n/128)*(k/128);
 gemm_a8w8_mxfp8_scale_small_lds_kernel<T><<<dim3(n/T::B_N,(m+T::B_M-1)/T::B_M),dim3(T::BLOCK_SIZE),T::lds_bytes(k),reinterpret_cast<hipStream_t>(stream)>>>(args);
 return hipGetLastError();
}
extern "C" __attribute__((visibility("default"))) int launch(int kid,const void* a,const void* b,const void* sfa,const void* sfb,void* c,int m,int n,int k,void* stream){
 if(kid==9046)return invoke<L9046>(a,b,sfa,sfb,c,m,n,k,stream);
 if(kid==9055)return invoke<L9055>(a,b,sfa,sfb,c,m,n,k,stream);
 return 1;
}
#endif
'''
    build=build[:start]+"def host_source():\n    return "+repr(host)+"\n"+build[end:]
    (OUT / 'build_and_audit.py').write_text(build)
    print(json.dumps({'status':manifest['status'],'out':str(OUT),'selected_symbols':len(manifest['selected_symbols'])}))


if __name__=='__main__':main()
