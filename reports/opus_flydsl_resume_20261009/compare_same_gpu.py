#!/usr/bin/env python3
"""Fixed selected OPUS versus all corresponding upstream FlyDSL configurations."""
from pathlib import Path
import argparse,ctypes,ctypes.util,gc,hashlib,json,math,os,random,statistics,sys,time
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
OLD=ROOT/'reports/opus_flydsl_comparison_20261008'
sys.path[:0]=[str(OLD/'python_packages'),str(ROOT),str(ROOT/'reports/opus_clang23_mixed_retune_20261008')]
import torch,flydsl
import aiter
from aiter.utility import dtypes
# AOT import skips top-level op imports; tuner helpers still require this alias.
aiter.dtypes=dtypes
from aiter.jit import core
from aiter.test_common import run_perftest
from aiter.utility.mp_tuner import checkAllclose
from aiter.ops.shuffle import shuffle_scale_blockscale_a,shuffle_scale_blockscale_b
from aiter.ops.flydsl.batched_gemm_a8w8_gfx950 import run_bmm_a8w8_mxfp8_gfx950,parse_bmm_kernel_name
from aiter.ops.flydsl.mxscale_preshuffle_kernels import run_gemm_a8w8_mxscale_preshuffle_gfx950
from csrc.opus_gemm import opus_gemm_mxscale_bpreshuffle_tune as tune
from csrc.opus_gemm.opus_gemm_common import kernels_list
from run_serial_tune import opus_ref,compare
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def normalize(v):
    d,b,df=v.lower().split(':');dev,f=df.split('.');return f'{int(d,16):04x}:{int(b,16):02x}:{int(dev,16):02x}.{int(f)}'
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--cases',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--rounds',type=int,default=6);args=ap.parse_args();assert not args.output.exists()
    plan=json.loads(args.plan.read_text());cases=json.loads(args.cases.read_text());module=core.get_module('module_deepgemm_opus');assert Path(module.__file__).resolve()==Path(plan['opus_library']).resolve();assert sha(module.__file__)==plan['opus_sha256'];assert flydsl.__version__=='0.3.4.1';assert str(OLD/'python_packages') in flydsl.__file__
    hip=ctypes.CDLL(ctypes.util.find_library('amdhip64') or '/opt/rocm/lib/libamdhip64.so');buf=ctypes.create_string_buffer(64);assert hip.hipDeviceGetPCIBusId(buf,len(buf),0)==0;bdf=normalize(buf.value.decode());assert bdf==normalize(os.environ['OPUS_EXPECTED_GPU_BDF'])
    convert,_,raw_stream,current_device=core._pybind_develop_hooks()
    result={'status':'running','started_unix':time.time(),'gpu_bdf':bdf,'gpu':str(torch.cuda.get_device_properties(0)),'pid':os.getpid(),'script_sha256':sha(__file__),'plan_sha256':sha(args.plan),'cases_sha256':sha(args.cases),'opus_sha256':sha(module.__file__),'flydsl_version':flydsl.__version__,'flydsl_path':flydsl.__file__,'method':'Fixed current all-backend OPUS selections and every matching upstream FlyDSL row. Same signed FP8 operands, exact logical E8M0 A1x128/B128x128 scale values; layout transforms prepared outside timing. Same GPU per shape, shared cloned pool8 plus reuse1, 3 randomized forward/reverse blocks/6 samples, original profiler 5warmup/51iterations. OPUS fixed workspace, FlyDSL official host API (including split-K reduction/counters/allocator GPU activities). Correctness against same independent FP32 reference, common external tolerance plus strict OPUS accumulation gate. Reference/shuffle never timed.','rows':[]}
    def save():
        p=args.output.with_suffix('.tmp');p.write_text(json.dumps(result,indent=2)+'\n');p.replace(args.output)
    save()
    for case in cases:
        start=time.monotonic();m,n,k,kid=[int(case[x]) for x in ['M','N','K','opus_kid']];split=int(kernels_list[kid].bpreshuffle_split_k)
        data=tune.generate_data(m,n,k,29,device='cuda:0');gen=torch.Generator(device='cuda:0').manual_seed(29)
        for name in ['x','w_reference']:
            value=data[name].float();sign=torch.randint(0,2,value.shape,device='cuda:0',generator=gen)*2-1;data[name]=(value*sign).to(data[name].dtype);del value,sign
        data['w']=tune.shuffle_weight(data['w_reference'],layout=(16,16))
        ref=opus_ref(*(data[x] for x in tune._REF_KEYS),with_bounds=True)
        # Native OPUS/BMM scales retain the original column-major A bytes.
        tensors=[data[x] for x in ['x','w','x_scale','w_scale','out']]
        mxpsh=any(x['kind']=='mxpsh' for x in case['flydsl'])
        if mxpsh:tensors += [shuffle_scale_blockscale_a(data['x_scale'],k),shuffle_scale_blockscale_b(data['w_scale'],n,k)]
        workspace=torch.empty(split*m*n,dtype=torch.float32,device='cuda:0') if split>1 else None
        if workspace is not None:tensors.append(workspace)
        footprint=sum(t.untyped_storage().nbytes() for t in tensors);free=torch.cuda.mem_get_info()[0];pool_count=max(1,min(8,int(free*.45/max(1,footprint))));assert pool_count>=2,(case,footprint,free)
        pool=[tensors]+[[t.clone(memory_format=torch.preserve_format) for t in tensors] for _ in range(pool_count-1)]
        # The BMM API requires a contiguous descriptor but interprets A-scale
        # bytes as column-major when x_scale_transposed=True. Reinterpret each
        # shared OPUS buffer's physical storage without changing its bytes.
        bmm_scale_pool=[]
        for values in pool:
            sa=values[2];physical=sa.T
            assert physical.is_contiguous(),(physical.shape,physical.stride())
            descriptor=physical.view(m,1,k//128)
            assert descriptor.is_contiguous() and descriptor.data_ptr()==sa.data_ptr()
            bmm_scale_pool.append(descriptor)
        method='opus_gemm_a8w8_blockscale_bpreshuffle_workspace_launch' if split>1 else 'opus_gemm_a8w8_blockscale_bpreshuffle_launch';launch=getattr(module,method)
        raw_pool=[tuple(convert(t.unsqueeze(0) if i in [0,1,4] else t) for i,t in enumerate(values[:5]))+(kid,)+((convert(values[-1]),) if split>1 else ()) for values in pool]
        def opus(index):
            module._set_current_hip_stream(raw_stream(current_device()));launch(*raw_pool[index]);return pool[index][4]
        variants=[('opus',opus)]
        for fi,frow in enumerate(case['flydsl']):
            name=frow['kernelName']
            if frow['kind']=='bmm':
                p=parse_bmm_kernel_name(name);assert p and p['splits']==int(frow['splitK'])
                def run(index,name=name):
                    a,b,sa,sb,out=pool[index][:5]
                    run_bmm_a8w8_mxfp8_gfx950(a.view(m,1,k),b.view(1,n,k),bmm_scale_pool[index],sb.view(1,n//128,k//128),out.view(m,1,n),kernel_name=name,x_scale_transposed=True);return out
            else:
                def run(index,name=name):
                    a,b,_,_,out,sa,sb=pool[index][:7];return run_gemm_a8w8_mxscale_preshuffle_gfx950(a,b,sa,sb,out,name)
            variants.append((f'flydsl_{fi}_{frow["kind"]}',run))
        checks={};invalid=[]
        for tag,fn in variants:
            try:
                data['out'].fill_(float('nan'));out=fn(0);torch.cuda.synchronize()
                assert bool(torch.isfinite(out).all().item()),(tag,'non-finite output')
                bound_err=compare(ref,out,printLog=False)
                if tag=='opus':assert bound_err==0,(case,bound_err)
                error=checkAllclose(out,ref[0].to(out.dtype),rtol=1e-2,atol=.01,tol_err_ratio=.05,printLog=False);assert math.isfinite(error) and error<=.05,(tag,error)
                checks[tag]={'errRatio':float(error),'accumulation_bound_errRatio':float(bound_err),'signed':True,'seed':29}
            except Exception as e:
                if tag=='opus':raise
                invalid.append({'variant':tag,'error':repr(e)});print(json.dumps({'shape':[m,n,k],'invalid':invalid[-1]}),flush=True)
        variants=[(tag,fn) for tag,fn in variants if tag in checks]
        assert len(variants)>=2,(case,invalid)
        entry={'case':case,'checks':checks,'invalid_flydsl':invalid,'pool_count':pool_count,'bytes_per_copy':footprint,'shared_addresses':[[t.data_ptr() for t in values] for values in pool],'records':[]};result['rows'].append(entry)
        del ref,out
        for round_id in range(args.rounds):
            block=round_id//2;order=[(tag,fn,p,count) for tag,fn in variants for p,count in [('reuse1',1),('pool8',pool_count)]]
            random.Random(20261008+block+m+n+k).shuffle(order)
            if round_id%2:order.reverse()
            for order_index,(tag,fn,protocol,count) in enumerate(order):
                index=0
                def timed():
                    nonlocal index
                    i=index%count;index+=1;return fn(i)
                out,us=run_perftest(timed,num_warmup=5,num_iters=51,num_rotate_args=1)
                assert math.isfinite(us) and us>0,(tag,protocol,us)
                entry['records'].append({'round':round_id,'order':order_index,'variant':tag,'protocol':protocol,'pool_count':count,'us':float(us)})
        entry['medians']={protocol:{tag:statistics.median(x['us'] for x in entry['records'] if x['protocol']==protocol and x['variant']==tag) for tag,_ in variants} for protocol in ['reuse1','pool8']}
        entry['seconds']=time.monotonic()-start;save();print(json.dumps({'shape':[m,n,k],'opus_kid':kid,'medians':entry['medians'],'seconds':entry['seconds'],'completed':len(result['rows']),'total':len(cases)}),flush=True)
        del timed,variants,checks,raw_pool,bmm_scale_pool,pool,tensors,data,workspace,run,fn,out,sa,physical,descriptor,values
        gc.collect();torch.cuda.empty_cache()
    assert sha(module.__file__)==plan['opus_sha256'];result.update(status='completed',ended_unix=time.time());save()
if __name__=='__main__':main()
