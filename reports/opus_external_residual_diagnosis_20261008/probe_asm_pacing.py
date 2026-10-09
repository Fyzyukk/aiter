#!/usr/bin/env python3
"""Preserve the full ASM operation while varying only host launch pacing."""
from pathlib import Path
import argparse,copy,ctypes,json,os,random,statistics,time
from probe_residual import OUT,ROOT,sha,core,tune,external_ref,checkAllclose,torch,profile_call
from aiter.utility.dtypes import aiter_tensor_t,torch_to_aiter
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();assert not args.output.exists()
    plan=json.loads(args.plan.read_text());name='module_gemm_a8w8_blockscale_bpreshuffle_asm';path=plan['libraries'][name];assert sha(path)==plan['library_sha256'][name]
    lib=ctypes.CDLL(path);entry=lib.gemm_a8w8_blockscale_bpreshuffle_asm
    entry.argtypes=[ctypes.POINTER(aiter_tensor_t)]*6+[ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.POINTER(aiter_tensor_t),ctypes.c_void_p];entry.restype=ctypes.c_int
    clear=lib.aiter_clear_last_error;clear.argtypes=[];clear.restype=None
    result={'status':'running','gpu_bdf':os.environ['OPUS_EXPECTED_GPU_BDF'],'owner_host_pid':os.environ.get('OPUS_OWNER_HOST_PID'),'started_unix':time.time(),'library_sha256':sha(path),'script_sha256':sha(__file__),'rows':[]}
    def save():
        p=args.output.with_suffix('.tmp');p.write_text(json.dumps(result,indent=2)+'\n');p.replace(args.output)
    save();tuner=tune.OpusMxscaleBpreshuffleTuner()
    for m,n,k,kid,split,name,up in [(240,768,7168,31,8,'_ZN5aiter42fp8gemm_bf16_blockscale_BpreShuffle_80x128E',10.8125),(64,6144,7168,20,5,'_ZN5aiter42fp8gemm_bf16_blockscale_BpreShuffle_64x128E',14.9877)]:
        task=next(t for t in tuner.get_gemm_a8w8_blockscale_asm_tune_task(('gfx950',256,m,n,k),True,0,True,{}) if t[0][3]==name and t[0][2]==split)
        _,generator,gen_args,func,call_args,_,ref_func,ref_args,ref_kwargs,_,*rest=task
        data=generator(*gen_args,device='cuda:0');reference=external_ref(*(data[x] for x in ref_args[0]),*ref_args[1:],**ref_kwargs)
        calls=tuple(data[x] for x in call_args[0])+tuple(call_args[1:]);pool=[calls]+[copy.deepcopy(calls) for _ in range(63)]
        packed=[];stream=ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        for values in pool:
            tensors=[torch_to_aiter(values[x]) for x in [0,1,4,2,3,5]]
            packed.append((tensors,tuple(ctypes.byref(x) for x in tensors[:5])+(None,split,name.encode(),1,ctypes.byref(tensors[5]),stream)))
        index_by_ptr={values[0].data_ptr():i for i,values in enumerate(pool)}
        def direct(*values):
            index=index_by_ptr[values[0].data_ptr()];clear();code=entry(*packed[index][1]);assert code==0,code;return values[4]
        def delayed(*values):
            end=time.perf_counter_ns()+20000
            while time.perf_counter_ns()<end:pass
            return direct(*values)
        def accuracy(out):
            torch.cuda.synchronize();err=checkAllclose(out,reference,rtol=rest[0],atol=rest[1],tol_err_ratio=.05,printLog=False);assert err<=.05;return float(err)
        for f in [func,direct,delayed]:data['out'].fill_(float('nan'));accuracy(f(*calls))
        variants=[('official',func,1),('direct',direct,1),('direct_delay20us',delayed,1),('official_pool64',func,64),('direct_pool64',direct,64)]
        row={'M':m,'N':n,'K':k,'kernelId':kid,'splitK':split,'kernelName':name,'us_upstream':up,'records':[]};result['rows'].append(row)
        for block in range(4):
            order=variants[:];random.Random(994+block).shuffle(order)
            for pass_id,seq in enumerate([order,list(reversed(order))]):
                for tag,f,count in seq:
                    out,details=profile_call(f,pool,count,False);row['records'].append({'variant':tag,'block':block,'pass':pass_id,'errRatio':accuracy(out),**details})
            save()
        row['summary']={tag:{key:statistics.median(x[key] for x in row['records'] if x['variant']==tag) for key in ['us','memset_us','gemm_us','gap_mean_us','gap_median_us']} for tag,_,_ in variants};save();print(json.dumps({'shape':[m,n,k],'summary':row['summary']}),flush=True)
        del pool,packed,data,reference,calls,out;torch.cuda.empty_cache()
    assert sha(path)==plan['library_sha256']['module_gemm_a8w8_blockscale_bpreshuffle_asm'];result.update(status='completed',ended_unix=time.time());save()
if __name__=='__main__':main()
