#!/usr/bin/env python3
from pathlib import Path
import argparse,collections,csv,hashlib,json,math
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
KEYS=['gfx','cu_num','M','N','K'];NEW={9001,9011}

def gm(values):return math.exp(sum(math.log(float(x)) for x in values)/len(values)) if len(values) else None

def best(df):return df.loc[df.groupby(KEYS,sort=False)['us'].idxmin()].sort_values(KEYS).reset_index(drop=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--partial',action='store_true');args=ap.parse_args()
    serial_queue=json.loads((OUT/'queue.json').read_text())
    serial_claims=[json.loads(x) for x in (OUT/'claim.jsonl').read_text().splitlines() if x.strip()]
    serial_completed={x['name'] for x in serial_claims if x.get('event')=='end' and x['returncode']==0 and not x['contamination']}
    eight_queue=json.loads((OUT/'queue_eight.json').read_text())
    eight_claims=[json.loads(x) for x in (OUT/'claim_eight.jsonl').read_text().splitlines() if x.strip()]
    queue={'commands':[c for c in serial_queue['commands'] if c['name'] in serial_completed]+eight_queue['commands']}
    claims=serial_claims+eight_claims
    completed={x['name'] for x in claims if x.get('event')=='end' and x['returncode']==0 and not x['contamination']}
    owners={};owner_errors=[];monitors=0
    for c in claims:
        if c.get('event')=='owner_identity':owners[c['name']]=c['host_pid'];current=c['name']
        if c.get('event')=='monitor':
            monitors+=1;name=c.get('name',current)
            for process in c['processes']:
                if process['pid']!=owners[name]:owner_errors.append({'command':name,'process':process})
    assert not owner_errors,owner_errors
    chunks=[];expected=[];runs=[]
    for item in queue['commands']:
        if item['name'] not in completed:continue
        meta_path=Path(item['argv'][item['argv'].index('--metadata')+1]);meta=json.loads(meta_path.read_text());assert meta['status']=='completed'
        p=meta_path.with_name(meta_path.stem+'_expected_opus_candidates.csv');expected.append(pd.read_csv(p))
        profile=Path(item['argv'][item['argv'].index('-o2')+1]);df=pd.read_csv(profile);df['checkpoint']=item['name'];chunks.append(df);runs.append(meta)
    assert chunks
    df=pd.concat(chunks,ignore_index=True);inventory=pd.concat(expected,ignore_index=True)
    observed=df[df.libtype.eq('opus')][['M','N','K','kernelId']]
    assert not observed.duplicated().any()
    assert set(map(tuple,observed.values))==set(map(tuple,inventory.values))
    df.to_csv(OUT/'profile.csv',index=False);inventory.to_csv(OUT/'expected_opus_candidates.csv',index=False)
    valid=df[df.us.gt(0)&df.us.map(math.isfinite)&df.errRatio.le(df.libtype.eq('opus').map({True:0.,False:.05}))].copy()
    opus=valid[valid.libtype.eq('opus')];old=best(opus[~opus.kernelId.isin(NEW)]);new=best(opus);external=best(valid[~valid.libtype.eq('opus')]);all_best=best(valid)
    for name,data in [('tuned_opus26.csv',old),('tuned_opus28.csv',new),('tuned_all.csv',all_best),('tuned_external.csv',external)]:data.drop(columns=['checkpoint']).to_csv(OUT/name,index=False)
    hist=pd.read_csv(ROOT/'reports/opus_current745_tables_20260930/shape_comparison_745.csv',encoding='utf-8-sig')
    compare=new[KEYS+['kernelId','us']].rename(columns={'kernelId':'opus28_kid','us':'opus28_us'})
    compare=compare.merge(old[KEYS+['kernelId','us']].rename(columns={'kernelId':'opus26_kid','us':'opus26_us'}),on=KEYS,validate='one_to_one')
    compare=compare.merge(external[KEYS+['libtype','kernelId','splitK','us']].rename(columns={k:'external_'+k for k in ['libtype','kernelId','splitK','us']}),on=KEYS,validate='one_to_one')
    compare=compare.merge(all_best[KEYS+['libtype','kernelId','splitK','us']].rename(columns={k:'all_'+k for k in ['libtype','kernelId','splitK','us']}),on=KEYS,validate='one_to_one')
    compare=compare.merge(hist[KEYS+['opus_kernelId','opus_us','same_run_baseline_us']],on=KEYS,validate='one_to_one')
    compare['incremental_speedup']=compare.opus26_us/compare.opus28_us;compare['incremental_latency_reduction_pct']=100*(1-compare.opus28_us/compare.opus26_us)
    compare['opus_vs_external_speedup']=compare.external_us/compare.opus28_us
    compare['historical_opus_time_change_pct']=100*(compare.opus28_us/compare.opus_us-1)
    compare['new_selected']=compare.opus28_kid.isin(NEW)
    compare.to_csv(OUT/'shape_comparison.csv',index=False)
    select=compare[compare.new_selected];select.to_csv(OUT/'new_candidate_selections.csv',index=False)
    counts=collections.Counter(int(k) for k in new.kernelId)
    candidate=[]
    for kid in sorted(inventory.kernelId.unique()):
        candidate.append({'kernelId':int(kid),'enumerated':int(inventory.kernelId.eq(kid).sum()),'valid':int(opus.kernelId.eq(kid).sum()),'opus_wins':counts[kid],'all_backend_wins':int((all_best.libtype.eq('opus')&all_best.kernelId.eq(kid)).sum())})
    pd.DataFrame(candidate).to_csv(OUT/'candidate_wins.csv',index=False)
    total=len(compare);original_shapes=pd.read_csv(OUT/'shapes_745.csv')
    summary={'status':'partial' if args.partial else 'completed','shapes':total,'expected_shapes':len(original_shapes),'completed_checkpoints':len(completed),'expected_checkpoints':len(queue['commands']),'candidate_ids':sorted(int(x) for x in inventory.kernelId.unique()),'profile_rows':len(df),'expected_opus_tasks':len(inventory),'valid_opus_tasks':len(opus),'invalid_by_backend':dict(collections.Counter(df.loc[~df.index.isin(valid.index),'libtype'])),'new_candidate_opus_wins':{str(k):int(new.kernelId.eq(k).sum()) for k in NEW},'new_candidate_all_backend_wins':{str(k):int((all_best.libtype.eq('opus')&all_best.kernelId.eq(k)).sum()) for k in NEW},'same_run_26_vs28':{'geomean_speedup':gm(compare.incremental_speedup),'time_sum26_us':float(compare.opus26_us.sum()),'time_sum28_us':float(compare.opus28_us.sum()),'time_sum_speedup':float(compare.opus26_us.sum()/compare.opus28_us.sum()),'improved_shapes':int(compare.incremental_speedup.gt(1).sum()),'improved_over1pct':int(compare.incremental_latency_reduction_pct.gt(1).sum()),'max_latency_reduction_pct':float(compare.incremental_latency_reduction_pct.max()),'selected_shapes_geomean_speedup':gm(select.incremental_speedup)},'opus28_vs_external':{'geomean_speedup':gm(compare.opus_vs_external_speedup),'opus_faster':int(compare.opus_vs_external_speedup.gt(1).sum()),'external_faster':int(compare.opus_vs_external_speedup.lt(1).sum())},'all_backend_counts':dict(collections.Counter(all_best.libtype)),'historical_descriptive_only':{'geomean_speedup':gm(compare.opus_us/compare.opus28_us),'faster':int(compare.opus28_us.lt(compare.opus_us).sum()),'slower':int(compare.opus28_us.gt(compare.opus_us).sum())},'owner_audit':{'strict_nonowner_pids':owner_errors,'monitor_samples':monitors,'owners':owners,'gpus':sorted({r['gpu_bdf'] for r in runs})},'large_rotation_counts':{k:sorted({x for r in runs for x in r['large_rotation_counts'].get(k,[])}) for k in {k for r in runs for k in r['large_rotation_counts']}},'limitations':['min of a single screening profile has selection noise; followup repeated trials required for small gains','same-run 26-vs28 measures added 9001/9011 only; existing 26 use their current accepted optimizations','historical Sept30 differences include software, GPU, clocks and timing changes','CK/CKTile/ASM FP32scale; OPUS native E8M0; common shuffle/reference outside timing','all backend sweep excludes Triton']}
    if not args.partial:
        assert total==745 and len(summary['candidate_ids'])==28
        assert set(map(tuple,compare[KEYS].values))==set(map(tuple,original_shapes[KEYS].values))
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
