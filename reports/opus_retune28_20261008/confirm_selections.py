#!/usr/bin/env python3
"""Five shuffled profiler rounds on one shared address pool per selected shape.

Retest the original top-three, both family members, and other candidates close
to the selected time. Native E8M0 inputs and exact official dispatch only.
"""
from pathlib import Path
import argparse,copy,gc,hashlib,json,math,random,statistics,sys,time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch
import pandas as pd
from csrc.opus_gemm import opus_gemm_mxscale_bpreshuffle_tune as tune
from aiter.jit import core
from aiter.test_common import run_perftest
OUT=Path(__file__).resolve().parent

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--selections',type=Path);args=ap.parse_args()
    plan=json.loads(args.plan.read_text());binary=Path(plan['official_binary']);assert hashlib.sha256(binary.read_bytes()).hexdigest()==plan['official_sha256']
    module=core.get_module('module_deepgemm_opus');assert Path(module.__file__).resolve()==binary.resolve()
    profile=pd.read_csv(OUT/'profile.csv');selections=pd.read_csv(args.selections or OUT/'new_candidate_selections.csv')
    result={'status':'running','started_unix':time.time(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'plan_sha256':hashlib.sha256(args.plan.read_bytes()).hexdigest(),'official_binary_sha256':plan['official_sha256'],'gpu':str(torch.cuda.get_device_properties(0)),'method':'5 randomized rounds; official profiler 5 warmup/51 iterations; identical shared 8-address pool; signed seed29 full accumulation interval validation','rows':[]}
    def save():
        tmp=args.output.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(args.output)
    save()
    for row in selections.itertuples(index=False):
        m,n,k=int(row.M),int(row.N),int(row.K)
        candidates=profile[profile.libtype.eq('opus')&profile.M.eq(m)&profile.N.eq(n)&profile.K.eq(k)&profile.us.gt(0)&profile.errRatio.eq(0)].sort_values('us')
        ids=set(int(v) for v in candidates.head(3).kernelId)
        ids.update(int(v) for v in candidates[candidates.us.le(float(row.opus28_us)*1.015)].kernelId)
        ids.update([int(row.opus26_kid),int(row.opus28_kid),9000 if int(row.opus28_kid)==9001 else 9010])
        # Preserve all legal close contenders, generally 3-6 candidates.
        shape_result={'shape':[m,n,k],'selected_id':int(row.opus28_kid),'screening_speedup':float(row.incremental_speedup),'candidate_ids':sorted(ids),'checks':{},'measurements':[]};result['rows'].append(shape_result)
        device=torch.device('cuda:0');data=tune.generate_data(m,n,k,29,device=device)
        gen=torch.Generator(device=device).manual_seed(29)
        for name in ['x','w_reference']:
            value=data[name].float();sign=torch.randint(0,2,value.shape,device=device,generator=gen)*2-1;data[name]=(value*sign).to(data[name].dtype)
        data['w']=tune.shuffle_weight(data['w_reference'],layout=(16,16))
        from run_serial_tune import opus_ref,compare
        reference=opus_ref(*(data[x] for x in tune._REF_KEYS),with_bounds=True)
        calls=tuple(data[x] for x in tune._BENCH_KEYS)
        for kid in sorted(ids):
            data['out'].fill_(float('nan'));actual=tune.run_bench(*calls,kid);torch.cuda.synchronize()
            err=compare(reference,actual,printLog=False)
            assert err==0,(m,n,k,kid,err)
            shape_result['checks'][str(kid)]={'errRatio':err,'seed':29,'signed':True}
        del reference
        footprint=sum(t.untyped_storage().nbytes() for t in calls)
        pool_count=max(1,min(8,int(torch.cuda.mem_get_info()[0]*.5/max(1,footprint))))
        pool=[tuple(copy.deepcopy(calls)) for _ in range(pool_count-1)]+[calls]
        def timed(kid):
            index=0
            def launch():
                nonlocal index
                arguments=pool[index%len(pool)];index+=1
                return tune.run_bench(*arguments,kid)
            return launch
        functions={kid:timed(kid) for kid in ids}
        # Each callable closes over the same pool, so deep-copying its empty
        # argument tuple creates no additional addresses or measured copies.
        for round_id in range(5):
            order=sorted(ids);random.Random(2900+round_id+sum([m,n,k])).shuffle(order)
            for order_id,kid in enumerate(order):
                _,us=run_perftest(functions[kid],num_warmup=5,num_iters=51,num_rotate_args=1)
                shape_result['measurements'].append({'round':round_id,'order_index':order_id,'kernelId':kid,'us':float(us)})
            save()
        medians={str(kid):statistics.median(x['us'] for x in shape_result['measurements'] if x['kernelId']==kid) for kid in ids}
        old_ids=ids-{9001,9011};old_kid=min(old_ids,key=lambda kid:medians[str(kid)]);new_kid=min(ids,key=lambda kid:medians[str(kid)])
        selected_id=int(row.opus28_kid)
        paired=[next(x['us'] for x in shape_result['measurements'] if x['round']==r and x['kernelId']==old_kid)/next(x['us'] for x in shape_result['measurements'] if x['round']==r and x['kernelId']==selected_id) for r in range(5)]
        shape_result.update(median_us=medians,old26_retested_best_id=old_kid,new28_retested_best_id=new_kid,selected_vs_old_median_speedup=medians[str(old_kid)]/medians[str(selected_id)],selected_paired_speedups=paired,selected_faster_rounds=sum(x>1 for x in paired),pool_count=pool_count)
        print(json.dumps({'shape':[m,n,k],'old_id':old_kid,'new_id':new_kid,'selected_id':selected_id,'speedup':shape_result['selected_vs_old_median_speedup'],'faster_rounds':shape_result['selected_faster_rounds']}),flush=True)
        save();del data,calls,pool,functions;gc.collect();torch.cuda.empty_cache()
    result['status']='completed';result['ended_unix']=time.time();save()

if __name__=='__main__':main()
