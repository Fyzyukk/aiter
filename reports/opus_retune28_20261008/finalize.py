from pathlib import Path
import hashlib,json,collections,math
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
KEYS=['gfx','cu_num','M','N','K'];NEW={9001,9011}

def gm(values):return math.exp(sum(math.log(x) for x in values)/len(values)) if len(values) else None

def audit(path):
    data=[json.loads(x) for x in path.read_text().splitlines()];owners={};errors=[];samples=0;end=[]
    for row in data:
        if row['event']=='owner_identity':owners[row['name']]=row['host_pid'];current=row['name']
        if row['event']=='monitor':
            samples+=1;name=row.get('name',current)
            errors += [{'name':name,'process':p} for p in row['processes'] if p['pid']!=owners[name]]
        if row['event']=='end':end.append({k:row[k] for k in ['name','returncode','contamination']})
    assert not errors
    return {'path':str(path),'strict_clean':not errors,'samples':samples,'owners':owners,'ends':end}

results=[json.loads(p.read_text()) for p in sorted(OUT.glob('confirm_shard_?.json'))];assert len(results)==8 and all(r['status']=='completed' for r in results)
rows=[r for record in results for r in record['rows']];assert len(rows)==78
records=[]
for r in rows:
    record={'gfx':'gfx950','cu_num':256,'M':r['shape'][0],'N':r['shape'][1],'K':r['shape'][2],'screen_selected_id':r['selected_id'],'screen_speedup':r['screening_speedup'],'retested_old_id':r['old26_retested_best_id'],'retested_new_id':r['new28_retested_best_id'],'selected_median_speedup':r['selected_vs_old_median_speedup'],'selected_faster_rounds':r['selected_faster_rounds'],'old_median_us':r['median_us'][str(r['old26_retested_best_id'])],'selected_median_us':r['median_us'][str(r['selected_id'])],'retested_new_median_us':r['median_us'][str(r['new28_retested_best_id'])],'pool_count':r['pool_count'],'candidate_ids':json.dumps(r['candidate_ids']),'paired_speedups':json.dumps(r['selected_paired_speedups'])}
    records.append(record)
confirm=pd.DataFrame(records).sort_values(KEYS).reset_index(drop=True);confirm.to_csv(OUT/'confirmation_comparison.csv',index=False)
checks=sum(len(r['checks']) for r in rows);measurements=sum(len(r['measurements']) for r in rows)
summary={}
for kid in sorted(NEW):
    sub=confirm[confirm.screen_selected_id.eq(kid)]
    summary[str(kid)]={'screen_selections':len(sub),'retest_selected_remains_best':int(sub.retested_new_id.eq(kid).sum()),'median_faster_than_old':int(sub.selected_median_speedup.gt(1).sum()),'geomean_selected_vs_retested_old':gm(sub.selected_median_speedup),'faster_all5_rounds':int(sub.selected_faster_rounds.eq(5).sum()),'slower_all5_rounds':int(sub.selected_faster_rounds.eq(0).sum()),'paired_win_count_histogram':dict(collections.Counter(int(x) for x in sub.selected_faster_rounds))}
# Useful reviewed alternative: revert unconfirmed new selections to the best
# original contender measured in the shared-pool confirmation. Timing columns
# explicitly identify their source; never mix these into screening metrics.
raw=pd.read_csv(OUT/'tuned_opus28.csv');all_raw=pd.read_csv(OUT/'tuned_all.csv');profile=pd.read_csv(OUT/'profile.csv')
reviewed=raw.copy();reviewed['timing_source']='screening_profiler'
reviewed_all=all_raw.copy();reviewed_all['timing_source']='screening_profiler'
for r in confirm.itertuples(index=False):
    chosen=r.retested_new_id;original=profile[profile.libtype.eq('opus')&profile.M.eq(r.M)&profile.N.eq(r.N)&profile.K.eq(r.K)&profile.kernelId.eq(chosen)].iloc[0]
    for target in [reviewed,reviewed_all]:
        mask=target.M.eq(r.M)&target.N.eq(r.N)&target.K.eq(r.K)
        for col in raw.columns:target.loc[mask,col]=original[col]
        new_time=r.retested_new_median_us;old_time=float(original.us);target.loc[mask,'us']=new_time
        target.loc[mask,'tflops']=float(original.tflops)*old_time/new_time;target.loc[mask,'bw']=float(original.bw)*old_time/new_time
        target.loc[mask,'timing_source']='five_round_shared_pool_profiler_median'
reviewed.to_csv(OUT/'tuned_opus28_rechecked.csv',index=False);reviewed_all.to_csv(OUT/'tuned_all_rechecked.csv',index=False)
# CSVs with the exact production schema can be consumed by explicit tuner
# replay; sidecar files carry the heterogeneous timing provenance.
reviewed.drop(columns=['timing_source']).to_csv(OUT/'tuned_opus28_rechecked_config.csv',index=False)
reviewed_all.drop(columns=['timing_source']).to_csv(OUT/'tuned_all_rechecked_config.csv',index=False)
identity_plan=json.loads((OUT/'plan.json').read_text());current_hash=hashlib.sha256((ROOT/'aiter/configs/model_configs/dsv4_a8w8_blockscale_bpreshuffle_tuned_gemm.csv').read_bytes()).hexdigest();assert current_hash==identity_plan['production_config_sha256']
audits=[audit(OUT/'claim.jsonl'),audit(OUT/'claim_eight.jsonl'),audit(OUT/'claim_confirm_eight.jsonl')];assert all(e['returncode']==0 and not e['contamination'] for a in audits[1:] for e in a['ends'])
final={'status':'completed','shape_count':745,'opus_candidates':28,'opus_tasks':13027,'correctness_all_opus_zero':True,'screening':json.loads((OUT/'summary.json').read_text()),'confirmation':{'shapes':78,'signed_correctness_checks':checks,'profiler_measurements':measurements,'rounds':5,'by_selected_candidate':summary,'rechecked_selections':{'9001':int(reviewed.kernelId.eq(9001).sum()),'9011':int(reviewed.kernelId.eq(9011).sum())},'method':'seed29 signed native E8M0, full accumulation bounds, 5 shuffled rounds/5warmup/51calls/shared eight-address pool; original top3 and close contenders, both family versions','limitation':'separate workload seed/address pool; all28 were screened but only close contenders retested; retest median wins are not necessarily all5 wins; no all745 GM inferred from mixed timing sources'},'owner_audits':audits,'production_config_unchanged':True,'production_config_sha256':current_hash,'new_kernel_bottleneck':'9001/9011 postchange primary bound remains unconfirmed; performance tuning does not constitute ATT/counter classification'}
(OUT/'final_review.json').write_text(json.dumps(final,indent=2)+'\n')
print(json.dumps({'confirmation':final['confirmation'],'production_config_unchanged':True},indent=2))
