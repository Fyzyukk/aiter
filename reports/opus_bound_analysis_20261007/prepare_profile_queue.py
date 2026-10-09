#!/usr/bin/env python3
"""CPU-only argv construction; root's run_when_idle owns GPU execution."""
import argparse
import json
from pathlib import Path

OUT=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,default=OUT/'plans/official_smoke.json')
    parser.add_argument('--targets',type=int,nargs='+',default=[2,17,18,24,26,27])
    parser.add_argument('--groups',nargs='+',default=['sq_ea','l2_tagmap','utcl1_credits','ta_lds'])
    parser.add_argument('--output',type=Path,default=OUT/'profile_queue.json')
    args=parser.parse_args()
    plan=json.loads(args.plan.read_text())
    evidence=json.loads((OUT/'counter_evidence.json').read_text())
    commands=[]
    for index in args.targets:
        if not 0<=index<len(plan['targets']): parser.error(f'target {index} out of range')
        target=plan['targets'][index]
        for name in args.groups:
            if name not in evidence['groups']: parser.error(f'unknown group {name}')
            counters=evidence['groups'][name]['aggregate_counters']
            label=f'target{index}_kid{target["kid"]}_{name}'
            outputdir=OUT/'profiles'/label
            command=['/opt/rocm/bin/rocprofv3','-E',str(OUT/'gfx950_extra_counters.yaml'),
                '--pmc',*counters,'--kernel-trace','--kernel-include-regex',
                'gemm_a8w8_mxfp8|opus_gemm_mxscale_bpreshuffle_reduce_kernel',
                '-d',str(outputdir),'-o',label,'--output-format','csv','--',
                '/opt/venv/bin/python3',str(OUT/'official_smoke.py'),'--plan',str(args.plan),
                '--output',str(outputdir/'application.json'),'--profile','--target-index',str(index),
                '--iters','51','--profile-rotation','0']
            parse=['/opt/venv/bin/python3',str(OUT/'parse_counters.py'),str(outputdir),
                '--output',str(outputdir/'parsed.json'),'--last','51']
            if target['kid'] in [9062,9063]: parse.append('--pair-split-k')
            commands.append({'name':label,'argv':command,'log':str(outputdir/'rocprof.log'),
                             'parse_after_success_argv':parse})
    result={'env':{'AITER_AOT_IMPORT':'1','AITER_JIT_DIR':str(OUT/'jit_baseline'),
        'GPU_ARCHS':'gfx950','CU_NUM':'256','OPUS_HIP_CLANG_PATH':'/root/toolchains/yuyzhang512-amdgpu-pin-op-dst-build/bin',
        'HIP_CLANG_PATH':'/root/toolchains/yuyzhang512-amdgpu-pin-op-dst-build/bin'},
        'precondition':'Run correctness smoke first. Each rocprof subprocess requires root physical-device idle lock; profile group runtime acceptance has not been verified.',
        'execution_notes':['run_when_idle uses HIP_VISIBLE_DEVICES from SMI hip_id and passes OPUS_EXPECTED_GPU_BDF.',
            'Commands are argv arrays and preserve one-pass/source identity.',
            'Profile auto storage rotation matches timing method; retain last51 per kernel.',
            'If an SDK group fails, save its error and split that group; do not claim unsupported counter results.',
            'Do not make speedup decisions using these instrumented durations.',
            'Add verified active --cu-count and --simd-count to parser when available; no global count inferred from env alone.'],
        'commands':commands}
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':'prepared_no_gpu_execution','commands':len(commands),'output':str(args.output)}))

if __name__=='__main__': main()
