#!/usr/bin/env python3
"""Current Clang23 CKTile timing protocols on one frozen binary and GPU."""
from pathlib import Path
import argparse, ctypes, ctypes.util, gc, hashlib, json, os, random, statistics, sys, time
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'reports/opus_clang23_mixed_retune_20261008'))
import torch
from aiter.jit import core
from aiter.test_common import run_perftest
from csrc.opus_gemm import opus_gemm_mxscale_bpreshuffle_tune as tune
from aiter.utility.mp_tuner import checkAllclose
from run_serial_tune import external_ref
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--case',type=Path,required=True);args=ap.parse_args()
    plan=json.loads(args.plan.read_text());case=json.loads(args.case.read_text());binary=Path(plan['libraries']['cktile'])
    assert sha(binary)==plan['library_sha256']['cktile']
    module=core.get_module('module_gemm_a8w8_blockscale_bpreshuffle_cktile_tune');assert Path(module.__file__).resolve()==binary.resolve()
    hip=ctypes.CDLL(ctypes.util.find_library('amdhip64') or '/opt/rocm/lib/libamdhip64.so');buf=ctypes.create_string_buffer(64);assert hip.hipDeviceGetPCIBusId(buf,len(buf),0)==0
    def norm(value):
        dom,bus,df=value.lower().split(':');dev,fun=df.split('.');return f'{int(dom,16):04x}:{int(bus,16):02x}:{int(dev,16):02x}.{int(fun)}'
    bdf=norm(buf.value.decode());assert bdf==norm(os.environ['OPUS_EXPECTED_GPU_BDF'])
    m,n,k,kid=map(int,[case['M'],case['N'],case['K'],case['kid']]);tuner=tune.OpusMxscaleBpreshuffleTuner()
    block_per_cu=sorted({candidate.BlockPerCu for candidate in tune.generic_tune.candidate_kernels_cktile_dict.values()})
    tasks=tuner.get_gemm_a8w8_blockscale_cktile_tune_task(('gfx950',256,m,n,k),True,0,True,block_per_cu,{'num_warmup':5,'num_iters':51})
    task=next(t for t in tasks if t[0][1]==kid)
    _,generator,gen_args,func,call_args,_,ref_func,ref_args,ref_kwargs,_,*rest=task
    data=generator(*gen_args,device='cuda:0');reference=external_ref(*(data[x] for x in ref_args[0]),*ref_args[1:],**ref_kwargs)
    calls=tuple(data[x] for x in call_args[0])+tuple(call_args[1:]);data['out'].fill_(float('nan'));out=func(*calls);torch.cuda.synchronize()
    error=checkAllclose(out,reference,rtol=rest[0],atol=rest[1],tol_err_ratio=.05,printLog=False);assert error<=.05,error
    protocols=[('rotate_5_51',0,5,51),('reuse_5_51',1,5,51),('rotate_50_200',0,50,200),('reuse_50_200',1,50,200)]
    result={'status':'running','started_unix':time.time(),'case':case,'gpu_bdf':bdf,'owner_host_pid':os.environ.get('OPUS_OWNER_HOST_PID'),'binary':str(binary),'binary_sha256':sha(binary),'script_sha256':sha(__file__),'method':'same current Clang23 binary/source/input data; 7 balanced forward/reverse blocks over 4 protocols; torch profiler; original automatic rotation for outputs<=32M, same full-tune cap8 for larger; external original FP32-scale data seed0/accuracy contract','accuracy_error_ratio':float(error),'records':[]}
    def save():
        tmp=args.output.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(args.output)
    save()
    for block in range(7):
        order=list(protocols);random.Random(277+block+sum([m,n,k])).shuffle(order)
        for pass_id,sequence in enumerate([order,list(reversed(order))]):
            for order_id,(name,rotation,warmup,iters) in enumerate(sequence):
                actual_rotation=rotation
                if rotation==0 and m*n>32*1024*1024:
                    footprint=sum(t.untyped_storage().nbytes() for t in calls if isinstance(t,torch.Tensor));actual_rotation=max(1,min(8,int(torch.cuda.mem_get_info()[0]*.5/max(1,footprint))))
                out,us=run_perftest(func,*calls,num_warmup=warmup,num_iters=iters,num_rotate_args=actual_rotation)
                error=checkAllclose(out,reference,rtol=rest[0],atol=rest[1],tol_err_ratio=.05,printLog=False);assert error<=.05
                result['records'].append({'block':block,'pass':pass_id,'order':order_id,'protocol':name,'requested_rotation':rotation,'actual_rotation_setting':actual_rotation,'warmup':warmup,'iters':iters,'us':float(us),'errRatio':float(error)})
            save()
    result['summary']={name:{'median_us':statistics.median(x['us'] for x in result['records'] if x['protocol']==name),'samples':[x['us'] for x in result['records'] if x['protocol']==name]} for name,_,_,_ in protocols}
    result.update(status='completed',ended_unix=time.time());assert sha(binary)==result['binary_sha256'];save();print(json.dumps(result['summary']),flush=True)
if __name__=='__main__':main()
