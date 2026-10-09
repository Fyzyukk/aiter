#!/usr/bin/env python3
"""Fixed ID17 CK A/B and full-call ASM event decomposition, all correctness gated."""
from pathlib import Path
import argparse, copy, ctypes, ctypes.util, hashlib, importlib.util, json, os, random, statistics, sys, time
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'reports/opus_clang23_mixed_retune_20261008')]
import pandas as pd
import torch
from aiter.jit import core
from aiter.test_common import get_trace_perf, post_process_data
from csrc.opus_gemm import opus_gemm_mxscale_bpreshuffle_tune as tune
from aiter.utility.mp_tuner import checkAllclose
from run_serial_tune import external_ref
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def profile_call(func,pool,count,graph_mode):
    warmup,iters=128,256
    for i in range(warmup):out=func(*pool[i%count])
    torch.cuda.synchronize()
    graph=None
    if graph_mode:
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for i in range(iters):out=func(*pool[i%count])
        graph.replay();torch.cuda.synchronize()
    for trace_attempt in range(4):
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],profile_memory=False,with_stack=False,with_modules=True) as prof:
            if graph_mode:graph.replay()
            else:
                for i in range(iters):out=func(*pool[i%count])
            torch.cuda.synchronize()
        events=[e for e in prof.events() if str(e.device_type).endswith('CUDA')]
        if len(events)>=iters:break
    assert len(events)>=iters,('incomplete GPU trace',len(events),graph_mode)
    official=float(get_trace_perf(prof,iters))
    cols=['name','self_cpu_time_total','self_device_time_total','device_type','device_index']
    df=pd.DataFrame([[getattr(e,x,None) for x in cols] for e in prof.events()],columns=cols)
    dropped,drop_count=post_process_data(df,iters,1); kept=df.drop(dropped)
    gpu=kept[kept.device_type.astype(str).str.contains('DeviceType.CUDA')]
    by_name=[]
    for name,group in gpu.groupby('name',sort=False):
        by_name.append({'name':name,'count':len(group),'us_per_call':float(group.self_device_time_total.sum()/(iters-drop_count))})
    assert abs(sum(x['us_per_call'] for x in by_name)-official)<1e-5,(by_name,official)
    memset=sum(x['us_per_call'] for x in by_name if 'memset' in x['name'].lower() or 'fill' in x['name'].lower())
    events=[e for e in prof.events() if str(e.device_type).endswith('CUDA')]
    starts=sorted([(e.time_range.start,e.time_range.end,e.name) for e in events])
    gaps=[max(0,b[0]-a[1]) for a,b in zip(starts,starts[1:])]
    assert official>0
    details={'us':official,'memset_us':memset,'gemm_us':official-memset,'event_groups':by_name,'kept_calls':iters-drop_count,'event_count':len(events),'trace_attempts':trace_attempt+1,'gap_median_us':statistics.median(gaps),'gap_mean_us':statistics.mean(gaps),'event_samples':[{'name':e.name,'us':e.self_device_time_total,'start_us':e.time_range.start,'end_us':e.time_range.end} for e in events[:8]]}
    return out,details
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--cases',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--focus',action='store_true');args=ap.parse_args()
    assert not args.output.exists()
    plan=json.loads(args.plan.read_text());cases=json.loads(args.cases.read_text());modules={}
    for name,path in plan['libraries'].items():
        assert sha(path)==plan['library_sha256'][name]
        if name.endswith('_asm'):modules[name]=ctypes.CDLL(path)
        elif name.startswith('module_ck_'):modules[name]=load_module(name,path)
        else:modules[name]=core.get_module(name);assert Path(modules[name].__file__).resolve()==Path(path).resolve()
    hip=ctypes.CDLL(ctypes.util.find_library('amdhip64') or '/opt/rocm/lib/libamdhip64.so');buf=ctypes.create_string_buffer(64);assert hip.hipDeviceGetPCIBusId(buf,len(buf),0)==0
    bdf=buf.value.decode();assert bdf.lower()==os.environ['OPUS_EXPECTED_GPU_BDF'].lower()
    result={'status':'running','started_unix':time.time(),'gpu_bdf':bdf,'owner_host_pid':os.environ.get('OPUS_OWNER_HOST_PID'),'plan_sha256':sha(args.plan),'script_sha256':sha(__file__),'library_sha256':plan['library_sha256'],'rows':[]}
    def save():
        tmp=args.output.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(args.output)
    save();tuner=tune.OpusMxscaleBpreshuffleTuner()
    for case in cases:
        m,n,k=[int(case[x]) for x in ['M','N','K']];backend=case['libtype'];kid=int(case['kernelId_new']);split=int(case['splitK_new']);shape=('gfx950',256,m,n,k)
        if backend=='ck':task=next(t for t in tuner.get_gemm_a8w8_blockscale_tune_task(shape,True,0,True,{}) if t[0][1]==kid)
        else:task=next(t for t in tuner.get_gemm_a8w8_blockscale_asm_tune_task(shape,True,0,True,{}) if t[0][3]==case['kernelName'] and t[0][2]==split)
        _,generator,gen_args,func,call_args,_,ref_func,ref_args,ref_kwargs,_,*rest=task
        data=generator(*gen_args,device='cuda:0');reference=external_ref(*(data[x] for x in ref_args[0]),*ref_args[1:],**ref_kwargs)
        calls=tuple(data[x] for x in call_args[0])+tuple(call_args[1:])
        if backend=='ck':calls=calls[:7]
        pool=[calls]+[copy.deepcopy(calls) for _ in range((16 if backend=='ck' else 64)-1)]
        def accuracy(out):
            torch.cuda.synchronize();err=checkAllclose(out,reference,rtol=rest[0],atol=rest[1],tol_err_ratio=.05,printLog=False);assert err<=.05,(case,err);return float(err)
        entry={'case':case,'records':[],'pool_count':len(pool),'tensor_bytes_per_copy':sum(x.nbytes for x in calls if isinstance(x,torch.Tensor))};result['rows'].append(entry)
        if backend=='ck':
            variants=[('official23',modules['module_gemm_a8w8_blockscale_bpreshuffle_tune'].gemm_a8w8_blockscale_bpreshuffle_tune),('current23',modules['module_ck_current23'].gemm_a8w8_blockscale_bpreshuffle_tune),('old3238_23',modules['module_ck_old3238'].gemm_a8w8_blockscale_bpreshuffle_tune),('old3383_23',modules['module_ck_old3383'].gemm_a8w8_blockscale_bpreshuffle_tune),('current20',modules['module_ck_current20'].gemm_a8w8_blockscale_bpreshuffle_tune)]
            if args.focus:
                variants=[(n,f) for n,f in variants if n in ['current23','old3383_23']]+[('current23_bf16off',modules['module_ck_current23_bf16off'].gemm_a8w8_blockscale_bpreshuffle_tune)]
            protocols=[('reuse_host',1,False),('pool16_host',16,False)]
        else:variants=[('asm',func)];protocols=[('reuse_host',1,False),('pool64_host',64,False)]
        for name,fn in variants:
            data['out'].fill_(float('nan'));assert accuracy(fn(*calls))<=.05
        combinations=[(name,fn,p,c,g) for name,fn in variants for p,c,g in protocols]
        for block in range(3):
            order=combinations[:];random.Random(811+block+sum([m,n,k])).shuffle(order)
            for pass_id,sequence in enumerate([order,list(reversed(order))]):
                for order_id,(name,fn,protocol,count,graph_mode) in enumerate(sequence):
                    out,details=profile_call(fn,pool,count,graph_mode)
                    record={'block':block,'pass':pass_id,'order':order_id,'variant':name,'protocol':protocol,'pool_count':count,'graph':graph_mode,'warmup':128,'iters':256,'errRatio':accuracy(out),**details};entry['records'].append(record)
            save()
        entry['summary']={f'{name}/{p}':{key:statistics.median(x[key] for x in entry['records'] if x['variant']==name and x['protocol']==p) for key in ['us','memset_us','gemm_us','gap_mean_us','gap_median_us']} for name,_ in variants for p,_,_ in protocols}
        save();print(json.dumps({'case':case,'summary':entry['summary']}),flush=True)
        del pool,data,calls,reference,out;torch.cuda.empty_cache()
    assert all(sha(path)==plan['library_sha256'][name] for name,path in plan['libraries'].items())
    result.update(status='completed',ended_unix=time.time());save()
if __name__=='__main__':main()
