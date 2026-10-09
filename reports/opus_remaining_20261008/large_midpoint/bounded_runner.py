#!/usr/bin/env python3
"""Large-output validation and 5 AB/BA complete-launch shared-pool Event.

References are computed in 256-row chunks outside measured events. A finite
eight-address pool bounds memory while preserving identical pointers/order
for both versions. Each graph contains 51 calls over those same addresses.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'reports/opus_bound_analysis_20261007'))
from experiment_runner import Runner,make_data,torch,tune,visible_gpu_identity,storage_guards
from official_smoke import OfficialRunner,check_large


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--check-only',action='store_true')
    ap.add_argument('--iters',type=int,default=51);ap.add_argument('--pool-count',type=int,default=8)
    ap.add_argument('--repetitions',type=int,default=2);args=ap.parse_args()
    plan=json.loads(args.plan.read_text());inv=json.loads((HERE.parent/'inventory.json').read_text())
    module=inv['official_module'];runners={}
    for label,path in plan['libraries'].items():
        runners[label]=(OfficialRunner(1,module['path'],module['sha256']) if Path(path).resolve()==Path(module['path']).resolve() else Runner(path))
    result={'status':'running','started':time.time(),'plan':plan,'plan_sha256':sha(args.plan),
            'libraries':{label:{'path':path,'sha256':sha(path)} for label,path in plan['libraries'].items()},
            'gpu':visible_gpu_identity(),'method':'5 AB/BA HIP-backed Event, complete captured launch',
            'reference_chunk_rows':256,'rows':[],'script_sha256':sha(__file__),
            'helper_sha256':{name:sha(ROOT/'reports/opus_bound_analysis_20261007'/name) for name in ['official_smoke.py','experiment_runner.py']}}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    def save():
        tmp=args.output.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(args.output)
    save()
    try:
        for index,target in enumerate(plan['targets']):
            kid=target['kid'];m,n,k=target['shape'];assert kid==9030
            data,guard=make_data(m,n,k,target.get('seed',17),target.get('signed',True))
            workspace=torch.empty((0,),device='cuda',dtype=torch.float32)
            calls=(*(data[name] for name in tune._BENCH_KEYS),workspace,kid)
            inputs=tuple(data[name] for name in tune._REF_KEYS)
            row={'kid':kid,'shape':[m,n,k],'target_index':index,'correctness':{},'measurements':[]};result['rows'].append(row)
            expected={}
            for label,runner in runners.items():
                first=None
                for _ in range(args.repetitions):
                    data['out'].fill_(float('nan'));out=runner(*calls);torch.cuda.synchronize()
                    check=check_large(inputs,out,guard,workspace,None,label,kid,(m,n,k))
                    if first is None:first=out.clone()
                    elif not torch.equal(first,out):raise RuntimeError(f'Nonrepeatable {label} {target}')
                expected[label]=first
                row['correctness'][label]={'repeatable':True,'repetitions':args.repetitions,**check}
            row['actual_official_module']=getattr(runners['baseline'],'actual_module',None)
            save()
            if not args.check_only:
                free=torch.cuda.mem_get_info()[0]
                footprint=sum(t.untyped_storage().nbytes() for t in calls[:6])
                count=max(1,min(args.pool_count,args.iters,int(free*.6/max(1,footprint))))
                pool=[copy.deepcopy(calls) for _ in range(count-1)]+[calls]
                row.update(pool_count=count,pool_bytes=sum(t.untyped_storage().nbytes() for a in pool for t in a[:6]),
                           pool_pointers=[[t.data_ptr() for t in a[:6]] for a in pool],iters=args.iters,shared_pool=True)
                stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream());graphs={}
                start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                start.record(stream);end.record(stream);end.synchronize()
                def reset():
                    for a in pool:a[2].fill_(float('nan'))
                def check_pool(label):
                    for j,a in enumerate(pool):
                        if not storage_guards(a[2],42) or not torch.equal(a[2],expected[label]):
                            raise RuntimeError(f'Event pool guard/repeat failure {label} {target} address{j}')
                    return {'checked_addresses':len(pool),'signed_reference_verified_expected':True,'guards_repeatability':True}
                for label,runner in runners.items():
                    with torch.cuda.stream(stream):
                        for a in pool:runner(*a)
                    stream.synchronize();graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph,stream=stream):
                        for j in range(args.iters):runner(*pool[j%len(pool)])
                    graphs[label]=graph
                    with torch.cuda.stream(stream):reset();graph.replay()
                    stream.synchronize();check_pool(label)
                for round_index in range(5):
                    order=list(runners) if round_index%2==0 else list(reversed(runners))
                    for label in order:
                        with torch.cuda.stream(stream):
                            reset();start.record(stream);graphs[label].replay();end.record(stream)
                        end.synchronize();us=start.elapsed_time(end)*1000/args.iters
                        assert us>0
                        measurement={'round':round_index,'label':label,'us_per_call':us,'order':order,**check_pool(label)}
                        row['measurements'].append(measurement);print(json.dumps({'kid':kid,'shape':[m,n,k],**measurement}),flush=True);save()
                row['median_us']={label:statistics.median(r['us_per_call'] for r in row['measurements'] if r['label']==label) for label in runners}
                row['median_speedup']=row['median_us']['baseline']/row['median_us']['candidate']
                del pool,graphs,graph,stream,start,end
            del data,guard,workspace,calls,inputs,expected,out,first
            torch.cuda.empty_cache();save()
        for label,identity in result['libraries'].items():assert sha(identity['path'])==identity['sha256']
        result['status']='passed';result['finished']=time.time();save()
    except BaseException as error:
        result.update(status='failed',error=str(error),finished=time.time());save();raise


if __name__=='__main__':main()
