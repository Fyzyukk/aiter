#!/usr/bin/env python3
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
OUT=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def statistics(frame):
    ratios=frame.clang23_over24_median;changes=(ratios-1)*100
    return {'shapes':len(frame),'geomean_latency_change_pct':float((np.exp(np.log(ratios).mean())-1)*100),'median_latency_change_pct':float(changes.median()),'clang23_slower':int((ratios>1).sum()),'clang23_faster':int((ratios<1).sum()),'within_5pct':int(changes.between(-5,5).sum()),'clang23_slower_over5pct':int(changes.gt(5).sum()),'clang23_faster_over5pct':int(changes.lt(-5).sum()),'within_1pct':int(changes.between(-1,1).sum())}
def main():
    allrows=[];raw=[];artifacts={}
    for path in sorted(OUT.glob('shard[0-7].json')):
        data=json.loads(path.read_text());assert data['status']=='completed';artifacts[path.name]=sha(path)
        assert len(data['pyinit_addresses'])==len(set(data['pyinit_addresses'].values()))==2
        for row in data['rows']:
            assert len(row['measurements'])==12
            assert all(x['errRatio']==0 for x in row['checks'].values())
            m,n,k=row['shape'];values={key:row[key] for key in ['kernelId','role','split_k','pool_count','clang23_over24_median','clang23_slower_rounds']}
            allrows.append({'M':m,'N':n,'K':k,**values,'clang23_median_us':row['median_us']['clang23'],'clang24_median_us':row['median_us']['clang24'],'clang23_latency_change_pct':(row['clang23_over24_median']-1)*100,'gpu_bdf':data['gpu_bdf'],'shard':path.stem})
            for point in row['measurements']:raw.append({'M':m,'N':n,'K':k,'kernelId':row['kernelId'],'role':row['role'],'gpu_bdf':data['gpu_bdf'],**point})
    allframe=pd.DataFrame(allrows);frame=allframe[allframe.role.eq('selected_unpinned')];control=allframe[allframe.role.eq('unchanged_pin_control')]
    assert len(frame)==589 and len(control)==16
    assert not allframe.duplicated(['M','N','K','kernelId']).any()
    allframe.to_csv(OUT/'compiler_ab_per_shape.csv',index=False);rawframe=pd.DataFrame(raw);rawframe.to_csv(OUT/'compiler_ab_raw.csv',index=False)
    assert len(rawframe)==605*12
    per_kid=pd.DataFrame([{'kernelId':kid,**statistics(rows)} for kid,rows in frame.groupby('kernelId')]);per_kid.to_csv(OUT/'compiler_ab_per_kid.csv',index=False)
    outliers=frame[frame.clang23_latency_change_pct.abs().gt(5)].sort_values('clang23_latency_change_pct',ascending=False);outliers.to_csv(OUT/'compiler_ab_outside_5pct.csv',index=False)
    round_stats=[]
    for (role,round_id),rows in rawframe.groupby(['role','round']):
        pivot=rows.pivot(index=['M','N','K','kernelId'],columns='compiler',values='us');ratios=pivot.clang23/pivot.clang24
        round_stats.append({'role':role,'round':int(round_id),'shapes':len(pivot),'geomean_clang23_latency_change_pct':float((np.exp(np.log(ratios).mean())-1)*100),'median_change_pct':float(((ratios-1)*100).median())})
    events=[json.loads(x) for x in (OUT/'claim_eight.jsonl').read_text().splitlines()];ends=[x for x in events if x['event']=='end'];assert len(ends)==8 and all(x['returncode']==0 and not x['contamination'] for x in ends);assert not any(x['event']=='external_work_started' for x in events)
    slow=outliers[outliers.clang23_latency_change_pct.gt(5)]
    summary={'status':'completed','scope':'all589 shapes whose latest OPUS-only winner is unpinned, using that fixed winner ID; 16 unchanged pin controls; not a new 745-shape tune or comparison of separately selected compiler winners','selected_unpinned':statistics(frame),'unchanged_pin_control':statistics(control),'rounds':round_stats,'stable_slower_over5pct_all6rounds':int(slow.clang23_slower_rounds.eq(6).sum()),'accuracy_checks':1210,'raw_measurements':len(rawframe),'physical_gpus':len(allframe.gpu_bdf.unique()),'ownership_monitor_samples':sum(x['event']=='monitor' for x in events),'gpu_children_returncode':[x['returncode'] for x in ends],'compiler_config_limitation':'same ROCm7.0 SDK and source/flags/host dispatch; clang23 uses own resource23, pin-clang24 requires ROCm resource20 headers; result compares these concrete compiler configurations, not all releases of version23 vs24','build_manifest_sha256':sha(OUT/'build_manifest.json'),'input_artifact_sha256':artifacts,'published_745_file_unchanged':True}
    (OUT/'compiler_ab_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2));print(per_kid.to_string(index=False))
if __name__=='__main__':main()
