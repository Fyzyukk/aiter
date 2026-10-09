#!/usr/bin/env python3
"""Same-process, same-address, balanced compiler A/B with signed accuracy checks."""
from pathlib import Path
import argparse, copy, ctypes, ctypes.util, gc, hashlib, importlib.util, json, math, os, random, statistics, sys, time
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'reports/opus_clang23_mixed_retune_20261008'))
import pandas as pd
import torch
from aiter.jit import core
from aiter.test_common import run_perftest
from csrc.opus_gemm import opus_gemm_mxscale_bpreshuffle_tune as tune
from csrc.opus_gemm.opus_gemm_common import kernels_list
from run_serial_tune import opus_ref, compare
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--selections',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    plan=json.loads(args.plan.read_text())
    libraries={tag:Path(path).resolve() for tag,path in plan['libraries'].items()}
    assert all(sha(path)==plan['library_sha256'][tag] for tag,path in libraries.items())
    hip=ctypes.CDLL(ctypes.util.find_library('amdhip64') or '/opt/rocm/lib/libamdhip64.so')
    buf=ctypes.create_string_buffer(64);assert hip.hipDeviceGetPCIBusId(buf,len(buf),0)==0
    def norm(value):
        domain,bus,df=value.lower().split(':');dev,fun=df.split('.')
        return f'{int(domain,16):04x}:{int(bus,16):02x}:{int(dev,16):02x}.{int(fun)}'
    bdf=norm(buf.value.decode());assert bdf==norm(os.environ['OPUS_EXPECTED_GPU_BDF'])
    module23=core.get_module('module_deepgemm_opus');assert Path(module23.__file__).resolve()==libraries['clang23']
    spec=importlib.util.spec_from_file_location('module_deepgemm_opus_ab24',libraries['clang24'])
    module24=importlib.util.module_from_spec(spec);spec.loader.exec_module(module24)
    assert Path(module24.__file__).resolve()==libraries['clang24']
    assert module23 is not module24
    assert module23.opus_gemm_a8w8_blockscale_bpreshuffle_launch is not module24.opus_gemm_a8w8_blockscale_bpreshuffle_launch
    handles={tag:ctypes.CDLL(str(libraries[tag])) for tag in ['clang23','clang24']}
    pointers={tag:ctypes.cast(getattr(handle,'PyInit_module_deepgemm_opus' if tag=='clang23' else 'PyInit_module_deepgemm_opus_ab24'),ctypes.c_void_p).value for tag,handle in handles.items()}
    assert len(set(pointers.values()))==2
    modules={'clang23':module23,'clang24':module24}
    convert,_,raw_stream,current_device=core._pybind_develop_hooks()
    selections=pd.read_csv(args.selections)
    result={'status':'running','started_unix':time.time(),'pid':os.getpid(),'owner_host_pid':os.environ.get('OPUS_OWNER_HOST_PID'),'gpu':str(torch.cuda.get_device_properties(0)),'gpu_bdf':bdf,'script_sha256':sha(__file__),'plan_sha256':sha(args.plan),'selections_sha256':sha(args.selections),'library_sha256':plan['library_sha256'],'pyinit_addresses':pointers,'library_maps':[line for line in Path('/proc/self/maps').read_text().splitlines() if any(str(p) in line for p in libraries.values())],'method':'6 rounds, 3 balanced AB/BA pairs; official torch profiler 5 warmup/51 iterations; same address pool reset to offset0 for each measurement; same kernel ID, same host objects; signed FP8/E8M0 reference gate on both libraries; fixed split-K uses shared preallocated workspace','rows':[]}
    def save():
        temporary=args.output.with_suffix('.tmp');temporary.write_text(json.dumps(result,indent=2)+'\n');temporary.replace(args.output)
    save()
    for row in selections.itertuples(index=False):
        start=time.monotonic();m,n,k,kid=map(int,[row.M,row.N,row.K,row.kernelId]);split=int(kernels_list[kid].bpreshuffle_split_k)
        data=tune.generate_data(m,n,k,29,device=torch.device('cuda:0'))
        gen=torch.Generator(device='cuda:0').manual_seed(29)
        for name in ['x','w_reference']:
            value=data[name].float();sign=torch.randint(0,2,value.shape,device='cuda:0',generator=gen)*2-1;data[name]=(value*sign).to(data[name].dtype)
            del value,sign
        data['w']=tune.shuffle_weight(data['w_reference'],layout=(16,16))
        reference=opus_ref(*(data[x] for x in tune._REF_KEYS),with_bounds=True)
        workspace=torch.empty(split*m*n,device='cuda:0',dtype=torch.float32) if split>1 else None
        tensors=[data['x'].unsqueeze(0),data['w'].unsqueeze(0),data['x_scale'],data['w_scale'],data['out'].unsqueeze(0)]
        if workspace is not None:tensors.append(workspace)
        def raw_arguments(values):
            args2=[convert(t) for t in values[:5]]+[kid]
            if len(values)>5:args2.append(convert(values[5]))
            return tuple(args2)
        raw=raw_arguments(tensors)
        checks={}
        method='opus_gemm_a8w8_blockscale_bpreshuffle_workspace_launch' if split>1 else 'opus_gemm_a8w8_blockscale_bpreshuffle_launch'
        for tag,module in modules.items():
            data['out'].fill_(float('nan'));module._set_current_hip_stream(raw_stream(current_device()));getattr(module,method)(*raw);torch.cuda.synchronize()
            err=compare(reference,data['out'],printLog=False);assert err==0,(m,n,k,kid,tag,err)
            checks[tag]={'errRatio':err,'seed':29,'signed':True}
        del reference,raw
        footprint=sum(t.untyped_storage().nbytes() for t in tensors)
        free=torch.cuda.mem_get_info()[0];pool_count=max(1,min(8,int(free*.5/max(1,footprint))))
        pool=[tensors]+[[t.clone(memory_format=torch.preserve_format) for t in tensors] for _ in range(pool_count-1)]
        raw_pool=[raw_arguments(values) for values in pool]
        entry={'shape':[m,n,k],'kernelId':kid,'role':row.role,'split_k':split,'checks':checks,'pool_count':pool_count,'pool_bytes':footprint*pool_count,'shared_addresses':[[t.data_ptr() for t in values] for values in pool],'measurements':[]};result['rows'].append(entry)
        for round_id in range(6):
            pair=round_id//2
            reverse=random.Random(230024+pair+sum([m,n,k])).randrange(2) ^ (round_id%2)
            order=['clang24','clang23'] if reverse else ['clang23','clang24']
            for order_index,tag in enumerate(order):
                module=modules[tag];launch=getattr(module,method);index=0
                def timed():
                    nonlocal index
                    arguments=raw_pool[index%pool_count];index+=1
                    module._set_current_hip_stream(raw_stream(current_device()));launch(*arguments)
                _,us=run_perftest(timed,num_warmup=5,num_iters=51,num_rotate_args=1)
                assert math.isfinite(us) and us>0
                entry['measurements'].append({'round':round_id,'order_index':order_index,'compiler':tag,'us':float(us)})
        medians={tag:statistics.median(x['us'] for x in entry['measurements'] if x['compiler']==tag) for tag in modules}
        ratios=[next(x['us'] for x in entry['measurements'] if x['round']==r and x['compiler']=='clang23')/next(x['us'] for x in entry['measurements'] if x['round']==r and x['compiler']=='clang24') for r in range(6)]
        entry.update(median_us=medians,clang23_over24_median=medians['clang23']/medians['clang24'],paired_ratios=ratios,clang23_slower_rounds=sum(x>1 for x in ratios),seconds=time.monotonic()-start)
        save();print(json.dumps({'shape':[m,n,k],'kid':kid,'role':row.role,'ratio23over24':entry['clang23_over24_median'],'completed':len(result['rows']),'total':len(selections)}),flush=True)
        del timed,launch,raw_pool,pool,tensors,data,workspace
        gc.collect();torch.cuda.empty_cache()
    result.update(status='completed',ended_unix=time.time());save()
if __name__=='__main__':main()
