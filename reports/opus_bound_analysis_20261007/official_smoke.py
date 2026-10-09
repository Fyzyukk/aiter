#!/usr/bin/env python3
"""Current official API correctness/timing and aligned private-baseline smoke.

Run only through root's physical idle-device lock. The prebuilt official JIT
must match current_build.json. Private libraries are selected by exact shape
and dispatch identity, not merely a common parent ID.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
os.environ.setdefault('AITER_AOT_IMPORT','1')
os.environ.setdefault('AITER_JIT_DIR',str(OUT/'jit_baseline'))

from experiment_runner import (Runner, check_result, guarded_workspace, make_data,
                              profile_arguments, run_perftest, storage_guards, torch,
                              tune, visible_gpu_identity)
from aiter.ops.opus import opus_gemm
from aiter.jit.core import get_module
from csrc.opus_gemm.opus_gemm_common import (A8W8_BPRESHUFFLE_TUNING_KIDS,
    a8w8_mxscale_gemm_bpreshuffle_kernels_list, a8w8_mxscale_bpreshuffle_supports_shape)

class OfficialRunner:
    def __init__(self,split,binary,binary_sha256):
        self.split=split
        self.expected_binary=Path(binary).resolve()
        self.expected_sha256=binary_sha256
        self.actual_module=None
    def __call__(self,x,w,out,x_scale,w_scale,workspace,kid):
        result=opus_gemm(x,w,out,kid=kid,layout='bpreshuffle',x_scale=x_scale,
            w_scale=w_scale,workspace=workspace.view(-1) if self.split>1 else None)
        self.last_workspace=workspace
        if self.actual_module is None:
            module=get_module('module_deepgemm_opus')
            path=Path(module.__file__).resolve()
            identity={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
            if path!=self.expected_binary or identity['sha256']!=self.expected_sha256:
                raise RuntimeError(f'Actual loaded official module differs from plan: {identity}, expected {self.expected_binary} {self.expected_sha256}')
            self.actual_module=identity
        return result

def check_large(reference_input,out,guard,workspace,workspace_guard,label,kid,shape):
    # Keep the 9030 >2GB-output reference bounded in memory. Each output row
    # chunk uses the unchanged independent FP32 interval contract.
    m,n,k=shape
    x,w,xscale,wscale=reference_input
    if m*n<=32*1024*1024:
        reference=tune.run_torch(x,w,xscale,wscale,with_bounds=True)
        result=check_result(reference,out,guard,workspace,workspace_guard,
            label=label,kid=kid,shape=shape)
        del reference
        return result
    for begin in range(0,m,256):
        end=min(begin+256,m)
        reference=tune.run_torch(x[begin:end],w,xscale[begin:end],wscale,with_bounds=True)
        error=tune.compare_outputs(reference,out[begin:end],printLog=False)
        del reference
        if error: raise RuntimeError(f'Numerical failure {label} {kid} rows{begin}:{end} shape{shape}: {error}')
    output_ok=storage_guards(out,42)
    workspace_ok=workspace_guard is None or (storage_guards(workspace,73) and bool(torch.isfinite(workspace).all()))
    if not output_ok or not workspace_ok: raise RuntimeError(f'Guard failure {label} {kid} shape{shape}')
    return {'errRatio':0,'output_guards':True,'workspace_guards':workspace_ok,'reference_chunk_rows':256}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,default=OUT/'plans/official_smoke.json')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--target-index',type=int)
    parser.add_argument('--kids',type=int,nargs='+')
    parser.add_argument('--rounds',type=int,default=1)
    parser.add_argument('--iters',type=int,default=11)
    parser.add_argument('--repetitions',type=int,default=2)
    parser.add_argument('--check-only',action='store_true')
    parser.add_argument('--profile',action='store_true')
    parser.add_argument('--profile-label',default='official')
    parser.add_argument('--profile-rotation',type=int,default=0)
    args=parser.parse_args()
    if min(args.rounds,args.repetitions)<1 or args.iters<2 or args.profile_rotation<0:
        parser.error('rounds/repetitions positive, iters>=2, profile-rotation>=0 required')
    if args.profile and args.target_index is None:
        parser.error('profile requires one --target-index; validate unprofiled first')
    plan=json.loads(args.plan.read_text())
    if args.target_index is not None and not 0<=args.target_index<len(plan['targets']):
        parser.error('target-index out of range')
    if plan['current_parent_count']!=len(A8W8_BPRESHUFFLE_TUNING_KIDS):
        raise RuntimeError('Official plan parent pool differs from current registry')
    binary=Path(plan['official_binary'])
    if binary.parent.resolve()!=Path(os.environ['AITER_JIT_DIR']).resolve():
        raise RuntimeError('AITER_JIT_DIR must select the recorded official JIT')
    if hashlib.sha256(binary.read_bytes()).hexdigest()!=plan['official_binary_sha256']:
        raise RuntimeError('Recorded official binary hash differs before launch')
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if head!=plan['source_head']: raise RuntimeError('Source HEAD differs from prepared official plan')
    result={'status':'running','started':time.time(),'source_head':head,
        'plan':plan,'official_binary_sha256':plan['official_binary_sha256'],
        'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'experiment_runner_sha256':hashlib.sha256((OUT/'experiment_runner.py').read_bytes()).hexdigest(),
        'profiling_only':args.profile,'timing':'complete official API/private launch; preallocated workspace; automatic argument rotation',
        'rows':[]}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    def save():
        tmp=args.output.with_suffix(args.output.suffix+'.tmp')
        tmp.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n'); tmp.replace(args.output)
    save()
    try:
        result['gpu']=visible_gpu_identity()
        for index,target in enumerate(plan['targets']):
            if args.target_index is not None and index!=args.target_index: continue
            kid=target['kid']; m,n,k=target['shape']
            if args.kids is not None and kid not in args.kids: continue
            instance=a8w8_mxscale_gemm_bpreshuffle_kernels_list[kid]
            if not a8w8_mxscale_bpreshuffle_supports_shape(instance,m,n,k):
                raise RuntimeError(f'Official support contract rejects {target}')
            split=instance.bpreshuffle_split_k
            data,guard=make_data(m,n,k,target.get('seed',17),target.get('signed',True))
            workspace,workspace_guard=(guarded_workspace(split,m,n) if split>1 else
                                      (torch.empty((0,),device='cuda',dtype=torch.float32),None))
            runners={'official':OfficialRunner(split,binary,plan['official_binary_sha256'])}
            identities={'official':{'path':str(binary),'sha256':plan['official_binary_sha256']}}
            for name,path in target.get('private_baselines',{}).items():
                runner=Runner(path,workspace=name in ['fine_wait','fine_n64'])
                if runner.split_count is not None and runner.split_count(kid)!=split:
                    raise RuntimeError(f'Private split mismatch {name} {target}')
                runners[name]=runner
                identities[name]={'path':path,'sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest()}
            call_args=(*(data[name] for name in tune._BENCH_KEYS),workspace,kid)
            ref_inputs=tuple(data[name] for name in tune._REF_KEYS)
            row={'target_index':index,'kid':kid,'shape':[m,n,k],'split':split,
                'purpose':target['purpose'],'libraries':identities,'correctness':{},'timings':[]}
            result['rows'].append(row)
            if args.profile:
                if args.profile_label not in runners: raise RuntimeError('Requested profile label does not match exact target')
                runner=runners[args.profile_label]
                rotated=profile_arguments(runner,call_args,args.iters,args.profile_rotation)
                for _ in range(5): runner(*call_args)
                torch.cuda.synchronize()
                for iteration in range(args.iters): runner(*rotated[iteration%len(rotated)])
                torch.cuda.synchronize()
                row.update(profile_label=args.profile_label,profile_iterations=args.iters,
                           profile_warmup=5,profile_rotation=len(rotated))
                del rotated
            else:
                for label,runner in runners.items():
                    first=None
                    for repetition in range(args.repetitions):
                        data['out'].fill_(float('nan'))
                        if split>1: workspace.fill_(float('nan'))
                        out=runner(*call_args); torch.cuda.synchronize()
                        checks=check_large(ref_inputs,out,guard,workspace,workspace_guard,label,kid,(m,n,k))
                        if first is None: first=out.clone()
                        elif not torch.equal(first,out): raise RuntimeError(f'Nonrepeatable {label} {target}')
                    row['correctness'][label]={'repetitions':args.repetitions,'repeatable':True,**checks}
                    del first
                if not args.check_only:
                    labels=list(runners)
                    for round_index in range(args.rounds):
                        order=labels if round_index%2==0 else list(reversed(labels))
                        for label in order:
                            data['out'].fill_(float('nan'))
                            if split>1: workspace.fill_(float('nan'))
                            out,us=run_perftest(runners[label],*call_args,num_warmup=5,num_iters=args.iters)
                            if not 0<us<float('inf'): raise RuntimeError(f'Invalid timing {us}')
                            checks=check_large(ref_inputs,out,guard,runners[label].last_workspace,workspace_guard,label,kid,(m,n,k))
                            row['timings'].append({'round':round_index,'label':label,'us':us,**checks})
                            print(json.dumps({'target_index':index,'kid':kid,'shape':[m,n,k],'round':round_index,'label':label,'us':us}),flush=True)
                        save()
                    row['median_us']={label:statistics.median(v['us'] for v in row['timings'] if v['label']==label) for label in labels}
                    row['private_over_official_us']={label:row['median_us'][label]/row['median_us']['official'] for label in labels if label!='official'}
            row['actual_official_module']=runners['official'].actual_module
            for label,identity in identities.items():
                if hashlib.sha256(Path(identity['path']).read_bytes()).hexdigest()!=identity['sha256']:
                    raise RuntimeError(f'Library mutated during official smoke: {label}')
            save()
            del data,guard,workspace,workspace_guard,call_args,ref_inputs,runners
            if 'out' in locals(): del out
            torch.cuda.empty_cache()
        result['status']='passed'
    except BaseException as error:
        result['status']='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed'
        result['error']={'type':type(error).__name__,'message':str(error)}
        raise
    finally:
        result['finished']=time.time(); save()

if __name__=='__main__': main()
