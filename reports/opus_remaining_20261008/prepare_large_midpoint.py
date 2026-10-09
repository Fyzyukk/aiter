#!/usr/bin/env python3
"""Prepare an isolated 9030 midpoint publication/retirement candidate; CPU only."""
from pathlib import Path
import difflib
import hashlib
import json
import shutil

OUT = Path(__file__).resolve().parent / 'large_midpoint'
ROOT = OUT.parents[2]
SOURCE = ROOT / 'csrc/opus_gemm/include'
HEADER = Path('gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_8wave_192x256_large_output_gfx950.cuh')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def modify(text):
    begin = text.index('    auto advance_tile = ')
    end = text.index('    // Prologue', begin)
    old = text[begin:end]
    new = '''    auto advance_tile = [&](auto stage_i, int tile_k) {
        constexpr int stage = decltype(stage_i)::value;
        constexpr int next_stage = (stage + 1) % T::NUM_STAGES;
        read_scales(tile_k + 1, v_sfa_next, v_sfb_next);
        // Overlap the first M repeat with the pending K+1 matrix tile.
        static_for<T::E_M>([&](auto m_i) {
            static_for<T::E_N>([&](auto n_i) {
                mma_scale_fragment(m_i, n_i);
                if constexpr (decltype(m_i)::value == T::E_M - 1)
                    load_b_fragment(n_i, number<next_stage>{});
            });
            if constexpr (decltype(m_i)::value == 0) {
                __builtin_amdgcn_sched_barrier(0);
                s_waitcnt_vmcnt(0_I);
                s_waitcnt_lgkmcnt(0_I);
                // Publish K+1 and retire all K readers before K+2 reuses its LDS slot.
                __builtin_amdgcn_s_barrier();
                if (tile_k + 2 < loops)
                    issue_matrix_prefetch(stage_i, tile_k + 2);
                __builtin_amdgcn_sched_barrier(0);
            }
            load_a_fragment(m_i, number<next_stage>{});
        });
        // Complete this wave's K+1 reads; the next midpoint protects matrix reuse.
        s_waitcnt_lgkmcnt(0_I);
        if (tile_k + 2 >= loops) {
            // C aliases matrix LDS: all final operands must be retired before C writes.
            __builtin_amdgcn_sched_barrier(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);
        }
        v_sfa = v_sfa_next;
        v_sfb = v_sfb_next;
    };

'''
    assert old.count('issue_matrix_prefetch') == new.count('issue_matrix_prefetch') == 1
    candidate = text[:begin] + new + text[end:]
    assert candidate[:begin] == text[:begin]
    assert candidate[candidate.index('    // Prologue'):] == text[end:]
    return candidate


LAUNCHER = '''#include <hip/hip_runtime.h>
#include <cstdint>
#define __HIPCC_RTC__ 1
#include "include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_8wave_192x256_large_output_gfx950.cuh"
extern "C" __attribute__((visibility("default")))
int launch(int kid,const void* a,const void* b,const void* sfa,const void* sfb,void* c,int m,int n,int k,void* stream) {
 constexpr int64_t input_limit=INT32_MAX;
 constexpr int64_t output_limit=INT64_MAX;
 if(kid!=9030||m<=0||n<=0||k<=0||k>16384||k%128||m%64||n%256||
    int64_t(m)>input_limit/k||int64_t(n)>input_limit/k||
    int64_t(m)<=(input_limit/2)/n||int64_t(m)>(output_limit/2)/n||
    int64_t(191)*n+256>input_limit/2||
    !a||!b||!sfa||!sfb||!c||reinterpret_cast<uintptr_t>(a)%16||
    reinterpret_cast<uintptr_t>(b)%16||reinterpret_cast<uintptr_t>(sfa)%16||
    reinterpret_cast<uintptr_t>(c)%16) return int(hipErrorInvalidValue);
 const uint64_t output_begin=reinterpret_cast<uintptr_t>(c);
 const uint64_t output_bytes=uint64_t(m)*n*2;
 const auto overlaps=[&](const void* p,uint64_t bytes) {
   const uint64_t begin=reinterpret_cast<uintptr_t>(p);
   return output_begin>=begin?output_begin-begin<bytes:begin-output_begin<output_bytes;
 };
 if(overlaps(a,uint64_t(m)*k)||overlaps(b,uint64_t(n)*k)||
    overlaps(sfa,uint64_t(m)*(k/128))||overlaps(sfb,uint64_t(n/128)*(k/128)))
   return int(hipErrorInvalidValue);
 opus_gemm_mxscale_bpreshuffle_kargs_gfx950 args{};
 args.ptr_a=a;args.ptr_b=b;args.ptr_sfa=sfa;args.ptr_sfb=sfb;args.ptr_c=c;
 args.m=m;args.n=n;args.k=k;args.batch=1;args.stride_a=k;args.stride_b=k;args.stride_c=n;
 args.stride_sfa=m;args.stride_sfb=k/128;args.stride_a_batch=m*k;args.stride_b_batch=n*k;
 args.stride_c_batch=0;args.stride_sfa_batch=m*(k/128);args.stride_sfb_batch=(n/128)*(k/128);
 using T=opus_gemm_mxscale_bpreshuffle_8wave_192x256_large_output_traits_gfx950;
 gemm_a8w8_mxfp8_scale_8wave_192x256_large_output_kernel<T>
   <<<dim3(n/256,(m+191)/192),512,0,reinterpret_cast<hipStream_t>(stream)>>>(args);
 return int(hipGetLastError());
}
'''


def main():
    assert not (OUT / 'source_manifest.json').exists(), 'Prepared candidate exists; preserve its evidence'
    inventory = json.loads((OUT.parent / 'inventory.json').read_text())
    entry = next(e for e in inventory['entries'] if e['parent_id'] == 9030)
    current = (SOURCE / HEADER).read_text()
    modified = modify(current)
    for side in ['baseline', 'candidate']:
        shutil.copytree(SOURCE, OUT / side / 'include')
        (OUT / side / 'launch.hip').write_text(LAUNCHER)
    (OUT / 'candidate/include' / HEADER).write_text(modified)
    diff = ''.join(difflib.unified_diff(current.splitlines(True), modified.splitlines(True),
                    fromfile='baseline/include/' + str(HEADER), tofile='candidate/include/' + str(HEADER)))
    (OUT / 'candidate.diff').write_text(diff)
    files = []
    for path in sorted((OUT / 'baseline/include').rglob('*')):
        if path.is_file():
            rel = path.relative_to(OUT / 'baseline')
            files.append({'path': str(rel), 'baseline_sha256': sha(path),
                          'candidate_sha256': sha(OUT / 'candidate' / rel)})
    changed = [f for f in files if f['baseline_sha256'] != f['candidate_sha256']]
    assert [f['path'] for f in changed] == ['include/' + str(HEADER)]
    contract = {
        'status': 'isolated_prepared_not_built', 'parents': [9030],
        'production_modified': False, 'shared_helpers_modified': False,
        'official_module': inventory['official_module'], 'source_files': files,
        'changed_headers': changed, 'diff_sha256': sha(OUT / 'candidate.diff'),
        'launcher_sha256': sha(OUT / 'candidate/launch.hip'), 'prepare_script_sha256': sha(__file__),
        'mechanism': 'Move matrix publication/retirement to after first M repeat, issue K+2 after that barrier; retain terminal C-alias retirement barrier.',
        'preserved': ['scale-first prologue including both barriers', 'loop and tile guards',
                      'MFMA order and scale bytes', 'matrix and scale addresses/layouts',
                      'C64 base, output row guard and output writeback', 'traits and shared helpers'],
        'synchronization_obligations': [
            'At the midpoint each wave has completed its K operand LDS reads; the cross-wave barrier retires K readers before K+2 overwrite.',
            'vmcnt(0) then lgkmcnt(0) before midpoint barrier publishes every wave K+1 async write before any K+1 matrix read.',
            'Per-wave terminal lgkmcnt(0) completes all K+1 operand reads; the following midpoint protects their stage from K+3 overwrite.',
            'Last advance retains a workgroup barrier after all final operand LDS reads, before any final-tile C staging can alias matrix LDS.',
            'loops=1 has no advance and retains the original post-operand prologue barrier; even/odd loop tails retain original predicates.'
        ],
        'readiness': 'Root CPU build and strict official FUNC/full-metadata/normalized-descriptor alignment required before numerical gate.'
    }
    (OUT / 'source_manifest.json').write_text(json.dumps(contract, indent=2) + '\n')
    shutil.copyfile(OUT.parent / 'scale_issue/event_runner.py', OUT / 'event_runner.py')
    base = {'kid': 9030, 'seed': 17, 'signed': True, 'private_baselines': {},
            **{k: entry[k] for k in ['symbol', 'kernel_function', 'traits', 'instruction_sha256', 'actual_configuration_ids']}}
    cases = [
        ([65536, 16384, 1536], 'Actual K1536 winner, same clean ATT target and M192 tail'),
        ([65536, 16384, 384], 'Legal same-M/N three-tile support control; not historical winner'),
        ([65536, 16384, 16384], 'Legal same-M/N max-K support control; not historical winner'),
    ]
    controls = [([16448, 65536, k], f'Legal M192 tail and loops={k//128}; C above signed32 byte extent')
                for k in [128, 256, 384, 512, 640]]
    for name, selected in [('screen', cases), ('tail_guard', controls)]:
        plan = {'name': 'large_midpoint_' + name, 'baseline': inventory['baseline'],
                'libraries': {'baseline': inventory['official_module']['path'],
                              'candidate': str(OUT / 'candidate/experiments.so')},
                'targets': [{**base, 'shape': shape, 'purpose': purpose} for shape, purpose in selected],
                'official_module': inventory['official_module'], 'candidate_not_built': True}
        (OUT / (name + '_plan.json')).write_text(json.dumps(plan, indent=2) + '\n')
    print(json.dumps({'status': contract['status'], 'changed_headers': len(changed), 'out': str(OUT)}))


if __name__ == '__main__':
    main()
