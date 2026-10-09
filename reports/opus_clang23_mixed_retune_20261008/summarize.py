"""Audit and summarize the fresh eight-GPU compiler-split sweep."""
from pathlib import Path
import collections,hashlib,json,math
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
KEYS=['gfx','cu_num','M','N','K'];CALL=KEYS+['libtype','kernelName','splitK']

def gm(values):return math.exp(sum(math.log(float(x)) for x in values)/len(values)) if len(values) else None

def stats(before,after):
 ratio=before/after
 return {'shapes':len(ratio),'geomean_speedup':gm(ratio),'faster':int(ratio.gt(1).sum()),'slower':int(ratio.lt(1).sum()),'faster_over5pct':int(ratio.gt(1.05).sum()),'slower_over5pct':int(ratio.lt(1/1.05).sum()),'median_latency_change_pct':float((100*(after/before-1)).median()),'before_time_sum_us':float(before.sum()),'after_time_sum_us':float(after.sum())}

def best(df):return df.loc[df.groupby(KEYS,sort=False)['us'].idxmin()].sort_values(KEYS).reset_index(drop=True)

def audit(path,expected):
 rows=[json.loads(line) for line in path.read_text().splitlines()];owners={};errors=[];ends=[];count=0
 for row in rows:
  if row['event']=='owner_identity':owners[row['name']]=row['host_pid'];current=row['name']
  if row['event']=='monitor':
   count+=1;name=row.get('name',current)
   errors.extend({'name':name,'process':p} for p in row['processes'] if p['pid']!=owners[name])
  if row['event']=='end':ends.append(row)
 assert not errors and len(ends)==expected and all(r['returncode']==0 and not r['contamination'] for r in ends)
 return {'strict_clean':True,'monitor_samples':count,'owners':owners,'ends':ends}

def main():
 plan=json.loads((OUT/'plan.json').read_text());queue=json.loads((OUT/'queue_eight.json').read_text());runs=[];chunks=[];inventories=[]
 owner_audit=audit(OUT/'claim_eight.jsonl',8)
 for item in queue['commands']:
  argv=item['argv'];meta=Path(argv[argv.index('--metadata')+1]);run=json.loads(meta.read_text());assert run['status']=='completed';runs.append(run)
  for name,path in plan['libraries'].items():
   assert run['libraries'][name]['sha256']==plan['library_sha256'][name]
   assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==plan['library_sha256'][name]
  df=pd.read_csv(Path(argv[argv.index('-o2')+1]));df['checkpoint']=item['name'];chunks.append(df)
  inventories.append(pd.read_csv(meta.with_name(meta.stem+'_expected_opus_candidates.csv')))
 raw=pd.concat(chunks,ignore_index=True);inventory=pd.concat(inventories,ignore_index=True)
 assert not raw.duplicated(KEYS+['libtype','kernelId','splitK']).any()
 observed=raw[raw.libtype.eq('opus')][['M','N','K','kernelId']]
 assert not observed.duplicated().any() and set(map(tuple,observed.values))==set(map(tuple,inventory.values))
 assert len(observed)==13027 and observed.kernelId.nunique()==28
 raw.to_csv(OUT/'profile.csv',index=False);inventory.to_csv(OUT/'expected_opus_candidates.csv',index=False)
 valid=raw[raw.us.gt(0)&raw.us.map(math.isfinite)&raw.errRatio.le(raw.libtype.eq('opus').map({True:0.,False:.05}))].copy()
 opus=best(valid[valid.libtype.eq('opus')]);external=best(valid[~valid.libtype.eq('opus')]);allbest=best(valid)
 columns=[c for c in raw.columns if c!='checkpoint']
 for name,df in [('tuned_opus.csv',opus),('tuned_external.csv',external),('tuned_all.csv',allbest)]:df[columns].to_csv(OUT/name,index=False)
 for backend in ['ck','cktile','asm']:best(valid[valid.libtype.eq(backend)])[columns].to_csv(OUT/('tuned_'+backend+'.csv'),index=False)
 assert len(opus)==len(external)==len(allbest)==745
 expected=pd.read_csv(OUT/'shapes_745.csv');assert set(map(tuple,allbest[KEYS].values))==set(map(tuple,expected[KEYS].values))
 old=ROOT/'reports/opus_retune28_20261008'
 compare=opus[KEYS+['kernelId','us']].rename(columns={'kernelId':'opus_kid','us':'opus_us'})
 for tag,df in [('external',external),('all',allbest),('old_opus',pd.read_csv(old/'tuned_opus28.csv')),('old_external',pd.read_csv(old/'tuned_external.csv')),('old_all',pd.read_csv(old/'tuned_all.csv'))]:
  compare=compare.merge(df[KEYS+['libtype','kernelId','splitK','us']].rename(columns={k:tag+'_'+k for k in ['libtype','kernelId','splitK','us']}),on=KEYS,validate='one_to_one')
 compare['opus_vs_external_speedup']=compare.external_us/compare.opus_us
 compare['old_vs_new_external_speedup']=compare.old_external_us/compare.external_us
 compare['old_vs_new_opus_speedup']=compare.old_opus_us/compare.opus_us
 compare['old_vs_new_all_speedup']=compare.old_all_us/compare.all_us
 compare.to_csv(OUT/'shape_comparison.csv',index=False)
 # Match the saved upstream call by backend/name/splitK. ASM IDs are ordinals.
 baseline=pd.read_csv(ROOT/'aiter/configs/model_configs/dsv4_a8w8_blockscale_bpreshuffle_tuned_gemm.csv');baseline=baseline[baseline.gfx.eq('gfx950')&baseline.cu_num.eq(256)]
 assert len(baseline)==745
 named=raw[raw.kernelName.ne('Null')&raw.kernelName.notna()]
 named=named.copy();named['identity_kid']=named.kernelId.where(~named.libtype.eq('asm'),-1)
 baseline=baseline.copy();baseline['identity_kid']=baseline.kernelId.where(~baseline.libtype.eq('asm'),-1)
 identity_call=CALL+['identity_kid']
 matched=baseline.merge(named,on=identity_call,how='left',suffixes=('_upstream','_new'),validate='one_to_one')
 matched['same_call_measured']=matched.us_new.notna()
 matched['same_call_valid']=matched.us_new.gt(0)&matched.us_new.map(lambda x:math.isfinite(x) if pd.notna(x) else False)&matched.errRatio_new.le(.05)
 matched['upstream_vs_new_speedup']=matched.us_upstream/matched.us_new
 matched.to_csv(OUT/'upstream_same_call_comparison.csv',index=False)
 matched[~matched.same_call_measured].to_csv(OUT/'upstream_calls_not_measured.csv',index=False)
 # CK B-preshuffle's wrapper accepts splitK but never passes KBatch to
 # blockwise_dispatch. Preserve CSV values while comparing executed calls.
 baseline_effective=baseline.copy();baseline_effective['effective_splitK']=baseline_effective.splitK.where(~baseline_effective.libtype.eq('ck'),0)
 named_effective=named.copy();named_effective['effective_splitK']=named_effective.splitK.where(~named_effective.libtype.eq('ck'),0)
 effective=baseline_effective.merge(named_effective,on=KEYS+['libtype','kernelName','effective_splitK','identity_kid'],how='left',suffixes=('_upstream','_new'),validate='one_to_one')
 effective['same_execution_measured']=effective.us_new.notna()
 effective['same_execution_valid']=effective.us_new.gt(0)&effective.us_new.map(lambda x:math.isfinite(x) if pd.notna(x) else False)&effective.errRatio_new.le(.05)
 effective['upstream_vs_new_speedup']=effective.us_upstream/effective.us_new
 effective.to_csv(OUT/'upstream_same_execution_comparison.csv',index=False)
 execution_stats={}
 for backend,part in effective.groupby('libtype'):
  measured=part[part.same_execution_valid]
  execution_stats[backend]={'upstream_rows':len(part),'same_execution_measured':int(part.same_execution_measured.sum()),'same_execution_valid':len(measured),**(stats(measured.us_upstream,measured.us_new) if len(measured) else {})}
 baseline_stats={}
 for backend,part in matched.groupby('libtype'):
  measured=part[part.same_call_valid]
  baseline_stats[backend]={'upstream_rows':len(part),'same_call_measured':int(part.same_call_measured.sum()),'same_call_valid':len(measured),'numeric_id_changed_but_same_call':int((part.kernelId_upstream.ne(part.kernelId_new)&part.same_call_measured).sum()),**(stats(measured.us_upstream,measured.us_new) if len(measured) else {})}
 oldraw=pd.read_csv(old/'profile.csv');oldnamed=oldraw[oldraw.kernelName.ne('Null')&oldraw.kernelName.notna()]
 oldnamed=oldnamed.copy();oldnamed['identity_kid']=oldnamed.kernelId.where(~oldnamed.libtype.eq('asm'),-1)
 paired=oldnamed.merge(named,on=identity_call,how='inner',suffixes=('_old','_new'),validate='one_to_one')
 paired=paired[paired.us_old.gt(0)&paired.us_new.gt(0)&paired.us_old.map(math.isfinite)&paired.us_new.map(math.isfinite)&paired.errRatio_old.le(paired.libtype.eq('opus').map({True:0.,False:.05}))&paired.errRatio_new.le(paired.libtype.eq('opus').map({True:0.,False:.05}))]
 paired['old_vs_new_speedup']=paired.us_old/paired.us_new;paired.to_csv(OUT/'old_new_same_call_comparison.csv',index=False)
 previous_backend_best={}
 for backend in ['ck','cktile','asm']:
  before=best(oldraw[oldraw.libtype.eq(backend)&oldraw.us.gt(0)&oldraw.us.map(math.isfinite)&oldraw.errRatio.le(.05)]);after=best(valid[valid.libtype.eq(backend)])
  joined=before.merge(after,on=KEYS,suffixes=('_old','_new'),validate='one_to_one');previous_backend_best[backend]=stats(joined.us_old,joined.us_new)
 counts=collections.Counter(int(kid) for kid in opus.kernelId)
 pd.DataFrame([{'kernelId':int(kid),'compiler_major':24 if kid in {9000,9001,9010,9011} else 23,'enumerated':int(inventory.kernelId.eq(kid).sum()),'valid':int(valid[valid.libtype.eq('opus')].kernelId.eq(kid).sum()),'opus_wins':counts[int(kid)],'all_backend_wins':int((allbest.libtype.eq('opus')&allbest.kernelId.eq(kid)).sum())} for kid in sorted(inventory.kernelId.unique())]).to_csv(OUT/'candidate_wins.csv',index=False)
 summary={'status':'completed','shapes':745,'opus_candidates':28,'expected_opus_tasks':13027,'valid_opus_tasks':int(valid.libtype.eq('opus').sum()),'profile_rows':len(raw),'raw_rows_by_backend':dict(collections.Counter(raw.libtype)),'invalid_by_backend':dict(collections.Counter(raw.loc[~raw.index.isin(valid.index),'libtype'])),'all_backend_counts':dict(collections.Counter(allbest.libtype)),'same_run_opus_vs_external':stats(compare.external_us,compare.opus_us),'old_new_best_descriptive_only':{'opus':stats(compare.old_opus_us,compare.opus_us),'external':stats(compare.old_external_us,compare.external_us),'all':stats(compare.old_all_us,compare.all_us)},'old_new_each_backend_best_descriptive_only':previous_backend_best,'upstream_same_call_descriptive_only':baseline_stats,'upstream_same_execution_descriptive_only':execution_stats,'ck_splitK_identity_correction':'Four CK saved splitK labels differed; CK B-preshuffle wrapper ignores this argument, so all409 original CK executions are matched.','old_new_same_call_descriptive_only':{backend:stats(part.us_old,part.us_new) for backend,part in paired.groupby('libtype')},'owner_audit':owner_audit,'gpu_bdfs':sorted({r['gpu_bdf'] for r in runs}),'compiler_policy':json.loads((OUT/'compiler_manifest.json').read_text()),'limitations':['Original upstream tune toolchain remains unconfirmed; this is a new Clang23 measured baseline.','ASM device binaries are precompiled; only its host wrapper is rebuilt.','CK/CKTile/ASM accept FP32 scales; OPUS accepts native E8M0; preserve each backend tuner contract.','No Triton candidates in this sweep.','Single screening profile minima include selection noise; historical timings are descriptive and are not mixed into fresh selections.','Timing alone does not establish a new compute/bandwidth/latency/dispatch bound.']}
 assert summary['valid_opus_tasks']==13027,summary['invalid_by_backend']
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({k:v for k,v in summary.items() if k not in ['owner_audit','compiler_policy']},indent=2))

if __name__=='__main__':main()
