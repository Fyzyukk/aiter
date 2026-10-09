#!/usr/bin/env python3
"""Extended64/128/256addresspool stability check; no result selection."""
from pathlib import Path
import argparse, copy, ctypes, ctypes.util, gc, hashlib, json, math, os, random, statistics, sys, time
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'reports/opus_clang23_mixed_retune_20261008')]
import torch
from aiter.jit import core
from aiter.test_common import run_perftest
from csrc.opus_gemm import opus_gemm_mxscale_bpreshuffle_tune as tune
from aiter.utility.mp_tuner import checkAllclose
from run_serial_tune import external_ref
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def normalize(value):
    dom,bus,df=value.lower().split(':');dev,fun=df.split('.')
    return f'{int(dom,16):04x}:{int(bus,16):02x}:{int(dev,16):02x}.{int(fun)}'
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--cases',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    assert not args.output.exists(),args.output
    plan=json.loads(args.plan.read_text());cases=json.loads(args.cases.read_text());loaded={}
    for name,path in plan['libraries'].items():
        assert sha(path)==plan['library_sha256'][name]
        if name.endswith('_asm'):
            # ASM uses the torch-free C ABI, not a PyInit extension.
            module=ctypes.CDLL(str(Path(path).resolve()))
        else:
            module=core.get_module(name);assert Path(module.__file__).resolve()==Path(path).resolve()
        loaded[name]=str(Path(path).resolve())
    hip=ctypes.CDLL(ctypes.util.find_library('amdhip64') or '/opt/rocm/lib/libamdhip64.so')
    buf=ctypes.create_string_buffer(64);assert hip.hipDeviceGetPCIBusId(buf,len(buf),0)==0
    bdf=normalize(buf.value.decode());assert bdf==normalize(os.environ['OPUS_EXPECTED_GPU_BDF'])
    result={'status':'running','started_unix':time.time(),'gpu_bdf':bdf,'gpu_properties':str(torch.cuda.get_device_properties(0)),'owner_host_pid':os.environ.get('OPUS_OWNER_HOST_PID'),'script_sha256':sha(__file__),'plan_sha256':sha(args.plan),'library_sha256':plan['library_sha256'],'mapped_libraries':[line for line in Path('/proc/self/maps').read_text().splitlines() if any(path in line for path in loaded.values())],'method':'same candidate/binary/data per shape; explicit64/128/256addresspools, constant256warmup/512iterations,3balancedforward/reverse blocks,6samples per pool. Inputs/outputs duplicated, all pools fully visited during warmup. All timing via original torch profiler; original FP32-scale inputs/reference/accuracy gate.','rows':[]}
    def save():
        temporary=args.output.with_suffix('.tmp');temporary.write_text(json.dumps(result,indent=2)+'\n');temporary.replace(args.output)
    save();tuner=tune.OpusMxscaleBpreshuffleTuner()
    for case in cases:
        m,n,k=int(case['M']),int(case['N']),int(case['K']);backend=case['libtype'];kid=int(case['kernelId_new']);split=int(case['splitK_new'])
        shape=('gfx950',256,m,n,k)
        if backend=='ck':
            tasks=tuner.get_gemm_a8w8_blockscale_tune_task(shape,True,0,True,{})
            task=next(t for t in tasks if t[0][1]==kid)
        elif backend=='asm':
            tasks=tuner.get_gemm_a8w8_blockscale_asm_tune_task(shape,True,0,True,{})
            task=next(t for t in tasks if t[0][3]==case['kernelName'] and t[0][2]==split)
        else:raise ValueError(backend)
        _,generator,gen_args,func,call_args,_,ref_func,ref_args,ref_kwargs,_,*rest=task
        data=generator(*gen_args,device='cuda:0');reference=external_ref(*(data[x] for x in ref_args[0]),*ref_args[1:],**ref_kwargs)
        calls=tuple(data[x] for x in call_args[0])+tuple(call_args[1:])
        def accuracy(out):
            torch.cuda.synchronize()
            error=checkAllclose(out,reference,rtol=rest[0],atol=rest[1],tol_err_ratio=.05,printLog=False)
            assert error<=.05,(case,error)
            return float(error)
        data['out'].fill_(float('nan'));error=accuracy(func(*calls))
        entry={'case':case,'initial_errRatio':error,'protocol_records':[],'ring_records':[]};result['rows'].append(entry)
        protocols=[('rotate_5_51',0,5,51),('reuse_5_51',1,5,51),('rotate_50_200',0,50,200),('reuse_50_200',1,50,200)]
        for block in range(0):
            order=list(protocols);random.Random(100827+block+sum([m,n,k])).shuffle(order)
            for pass_id,sequence in enumerate([order,list(reversed(order))]):
                for order_id,(name,rotation,warmup,iters) in enumerate(sequence):
                    out,us=run_perftest(func,*calls,num_warmup=warmup,num_iters=iters,num_rotate_args=rotation)
                    assert math.isfinite(us) and us>0
                    entry['protocol_records'].append({'block':block,'pass':pass_id,'order':order_id,'protocol':name,'rotation_setting':rotation,'warmup':warmup,'iters':iters,'us':float(us),'errRatio':accuracy(out)})
            save()
        entry['protocol_summary']={}
        torch.cuda.empty_cache()
        pool=[calls]+[tuple(copy.deepcopy(calls)) for _ in range(255)]
        tensor_bytes=sum(t.untyped_storage().nbytes() for t in calls if isinstance(t,torch.Tensor))
        read_keys=list(call_args[0]);input_bytes=sum(data[name].untyped_storage().nbytes() for name in read_keys if name not in ['out','zero_bias'])
        assert all(len({values[i].data_ptr() for values in pool})==256 for i,v in enumerate(calls) if isinstance(v,torch.Tensor))
        entry['ring_pool_bytes_per_copy']=tensor_bytes;entry['ring_input_bytes_per_copy']=input_bytes;entry['ring_input_addresses']=[[t.data_ptr() for t in values if isinstance(t,torch.Tensor)] for values in pool]
        for block in range(3):
            order=[64,128,256];random.Random(641+block+sum([m,n,k])).shuffle(order)
            for pass_id,sequence in enumerate([order,list(reversed(order))]):
                for order_id,count in enumerate(sequence):
                    index=0
                    def timed():
                        nonlocal index
                        values=pool[index%count];index+=1
                        return func(*values)
                    out,us=run_perftest(timed,num_warmup=256,num_iters=512,num_rotate_args=1)
                    entry['ring_records'].append({'block':block,'pass':pass_id,'order':order_id,'pool_count':count,'pool_bytes':count*tensor_bytes,'input_bytes':count*input_bytes,'warmup':256,'iters':512,'us':float(us),'errRatio':accuracy(out)})
            save()
        entry['ring_summary']={str(count):{'median_us':statistics.median(x['us'] for x in entry['ring_records'] if x['pool_count']==count),'samples':[x['us'] for x in entry['ring_records'] if x['pool_count']==count]} for count in [64,128,256]}
        save();print(json.dumps({'case':case,'protocol_us':{tag:x['median_us'] for tag,x in entry['protocol_summary'].items()},'ring_us':{tag:x['median_us'] for tag,x in entry['ring_summary'].items()}}),flush=True)
        del timed,pool,data,calls,reference,out
        gc.collect();torch.cuda.empty_cache()
    assert all(sha(path)==plan['library_sha256'][name] for name,path in plan['libraries'].items())
    result.update(status='completed',ended_unix=time.time());save()
if __name__=='__main__':main()
