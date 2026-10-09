#!/usr/bin/env python3
"""Validate completed diagnostics and preserve exact evidence separately from tune."""
from pathlib import Path
import csv,hashlib,json,statistics
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def csv_write(name,rows):
    with (OUT/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    raw=[];ck=[];asm=[];pacing=[];files=[]
    for prefix in ['shard','focus','pacing']:
        for i in range(8):
            path=OUT/f'{prefix}{i}.json';r=json.loads(path.read_text());assert r['status']=='completed'
            files.append({'path':str(path),'sha256':sha(path),'gpu_bdf':r['gpu_bdf'],'records':sum(len(x['records']) for x in r['rows'])})
            for entry in r['rows']:
                case=entry.get('case',entry)
                for record in entry['records']:
                    assert 0<record['us']<1e6 and 0<=record['errRatio']<=.05
                    assert abs(sum(x['us_per_call'] for x in record['event_groups'])-record['us'])<1e-5
                    raw.append({'run':prefix,'shard':i,'gpu_bdf':r['gpu_bdf'],'M':int(case['M']),'N':int(case['N']),'K':int(case['K']),'backend':case.get('libtype','asm'),'variant':record['variant'],'protocol':record.get('protocol',record['variant']),'block':record['block'],'pass':record['pass'],'us':record['us'],'memset_us':record['memset_us'],'gemm_us':record['gemm_us'],'errRatio':record['errRatio'],'gap_mean_us':record['gap_mean_us'],'event_count':record['event_count'],'kept_calls':record['kept_calls']})
                if case.get('libtype')=='ck':
                    s=entry['summary'];row={'run':prefix,'gpu_bdf':r['gpu_bdf'],'M':int(case['M']),'N':int(case['N']),'K':int(case['K']),'kernelId':17,'splitK':0,'upstream_us':case['us_upstream'],'full_tune_us':case['us_new']}
                    for protocol in ['reuse_host','pool16_host']:
                        old=s['old3383_23/'+protocol]['us'];now=s['current23/'+protocol]['us']
                        row['current23_'+protocol+'_us']=now;row['old3383_23_'+protocol+'_us']=old;row['new_ck_vs_old_ck_'+protocol+'_pct']=(now/old-1)*100
                    if prefix=='focus':
                        for protocol in ['reuse_host','pool16_host']:row['current23_bf16off_'+protocol+'_us']=s['current23_bf16off/'+protocol]['us']
                    else:
                        for protocol in ['reuse_host','pool16_host']:row['current20_'+protocol+'_us']=s['current20/'+protocol]['us'];row['official23_'+protocol+'_us']=s['official23/'+protocol]['us']
                    ck.append(row)
                elif prefix in ['shard','focus']:
                    for tag,v in entry['summary'].items():asm.append({'run':prefix,'gpu_bdf':r['gpu_bdf'],'M':int(case['M']),'N':int(case['N']),'K':int(case['K']),'kernelId':int(case['kernelId_new']),'splitK':int(case['splitK_new']),'upstream_us':case['us_upstream'],'protocol':tag,**v})
                else:
                    for tag,v in entry['summary'].items():pacing.append({'gpu_bdf':r['gpu_bdf'],'M':entry['M'],'N':entry['N'],'K':entry['K'],'kernelId':entry['kernelId'],'splitK':entry['splitK'],'upstream_us':entry['us_upstream'],'variant':tag,**v})
    csv_write('valid_raw.csv',raw);csv_write('ck_ab.csv',[r for r in ck if r['run']=='shard']);csv_write('ck_bf16_ab.csv',[r for r in ck if r['run']=='focus']);csv_write('asm_components.csv',asm);csv_write('asm_pacing.csv',pacing)
    expected={'aiter/configs/model_configs/dsv4_a8w8_blockscale_bpreshuffle_opus_tuned_gemm.csv':'3c54663129807c1ff264b4be3c1265b75bc603b85f87b252897759a84b3f0f01','reports/opus_clang23_mixed_retune_20261008/profile.csv':'f156e72850256713f0aa56b9398735300ba675c88ffa89f5977ee791e61a28a1','aiter/configs/model_configs/dsv4_a8w8_blockscale_bpreshuffle_tuned_gemm.csv':'6005999f566cf303ec8622575a1b18a6d61840af499e2cbf97421f01615ba724'}
    actual={p:sha(ROOT/p) for p in expected};assert actual==expected
    owner_logs=[]
    for name in ['claim_eight.jsonl','claim_retry.jsonl','claim_focus.jsonl','claim_pacing.jsonl']:
        records=[json.loads(line) for line in (OUT/name).read_text().splitlines()];assert not any(x['event']=='external_work_started' for x in records)
        owner_logs.append({'name':name,'owners':len([x for x in records if x['event']=='owner_identity']),'successful_exits':len([x for x in records if x['event']=='end' and x['returncode']==0]),'failed_exits':[x for x in records if x['event']=='end' and x['returncode']!=0],'foreign_processes':False})
    assert len(raw)==1600
    summary={'valid_timing_accuracy_records':len(raw),'max_errRatio':max(x['errRatio'] for x in raw),'files':files,'preserved_tune_sha256':actual,'owner_logs':owner_logs,'method':'Fixed CK ID17, identical generated TU/current common wrapper, historical CK33b62/83566 versus currentaf9, same Clang23 SDK/flags, direct fixed-ID host plus official host control; same GPU and pool; 3 balanced forward/reverse blocks, 6samples/config, 128warmup/256iters. Second build isolates builtin BF16 only. ASM full operation includes zero-init; profiler filtered device event sums decomposed into Memset and GEMM. Third run fixes CABI descriptors and changes host pacing, 8GPUs independently paired, 4balancedblocks/8samples.','exclusions':'Import/argument/graph-trace attempts retained for audit but excluded. HIP graph profiler produced missing events and one process crash; graph results are not used in conclusions. One host shard encountered stale diagnostic output and was rerun to completion; no tune output changed.','limitations':'Historical compiler/container/runtime clock/power/actual binary and timing receipts unknown; CK version-level effect verified for ID17, not all409 CK shapes or a single source change. ASM host-pacing effect verified without proving which historical protocol was used; device gaps not included in reported us.'}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({'records':len(raw),'errRatio_max':summary['max_errRatio'],'preserved':actual},indent=2))
if __name__=='__main__':main()
