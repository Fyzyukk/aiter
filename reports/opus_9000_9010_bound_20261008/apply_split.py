"""Finalize and apply the small optional candidate set after clean formal API."""
from pathlib import Path
import datetime, hashlib, json, math, statistics, subprocess

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
HERE=OUT/'split_formal_scale_reset'
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
write=lambda p,d:Path(p).write_text(json.dumps(d,indent=2)+'\n')
identity=read(HERE/'identity_audit.json');api=read(HERE/'api_results.json');owner=read(OUT/'owner_audit.json')
def clean(log,name):
    r=next(r for r in owner['records']if Path(r['path']).resolve()==(OUT/log).resolve())
    e=next(e for e in r['epochs']if e['name']==name)
    assert not r['incomplete'] and e['strict_clean'] and e['monitor_count']>0
    return {'path':r['path'],'sha256':r['sha256'],'epoch':e}
api_owner=clean('split_scale_api_claim.jsonl','split_scale_reset_formal_API')
assert api['status']=='passed' and len(api['rows'])==14
assert api['official_binary_sha256']==identity['official_module']['sha256']==sha(identity['official_module']['path'])
for r in api['rows']:
    c=r['correctness']['official']
    assert c['repetitions']==8 and c['errRatio']==0 and c['repeatable'] and c['output_guards'] and c['workspace_guards']
    assert r['actual_official_module']['sha256']==identity['official_module']['sha256']

def event_rows(path):
    d=read(path);assert d['status']=='passed';rows=[]
    for r in d['rows']:
        e=r.get('event_confirmation',r);by={(v['round'],v['label']):v['us_per_call']for v in e['measurements']}
        ratio=e['median_speedup'];ratio=ratio['candidate']if isinstance(ratio,dict)else ratio
        assert math.isclose(ratio,e['median_us']['baseline']/e['median_us']['candidate'])
        pairs=[by[(i,'baseline')]/by[(i,'candidate')]for i in range(5)]
        rows.append({'shape':r['shape'],'speedup':ratio,'faster_rounds':sum(v>1 for v in pairs),'pairs':pairs,
                     'median_us':e['median_us'],'source':str(path.relative_to(OUT))})
    return rows
confirm=event_rows(HERE/'confirm9000_event.json')
confirm_owner=clean('split_reset_confirm_claim.jsonl','split9000_scale_reset_repeat')
full9000=read(OUT/'scale_reset/full9000_analysis.json')
full9010=read(OUT/'loop_unroll4/full9010_analysis.json')
assert full9000['status']=='reject_9000_global_scale_reset_clean_serial'
assert full9010['geomean_speedup']>1 and full9010['stable_losers']==0
repeat_shapes=[[1792,7168,384],[2048,7168,3072]]
for shape in repeat_shapes:
    before=next(r for r in full9000['rows']if r['shape']==shape)
    after=next(r for r in confirm if r['shape']==shape)
    assert before['faster_rounds']==5 and before['speedup']>1 and after['speedup']>1
new_candidates=[]
for r in identity['new_candidates']:
    row=dict(r);kid=r['kid']
    if kid==9001:
        row.update(decision_reason='保留为可选tuner候选：两原5/5正样本在新seed仍median微正；全129 GM略负，禁止全局替换或宣称稳定普遍收益。',
                   decision='retain_optional_only_weak_finite_evidence',
                   performance={'global_full129':{k:full9000[k]for k in ['completed_shapes','geomean_speedup','positive_medians','stable_losers']},
                                'original_five_of_five_positive_shapes':repeat_shapes,'independent_seed29_repeat':confirm,
                                'repeat_owner':confirm_owner,'full129_path':str(OUT/'scale_reset/full9000_analysis.json')},
                   caveat='新seed原两正样本仅3/5轮快，median+0.1114%/+0.0387%；收益处于噪声敏感区。编译资源下降本身不是性能收益或dispatch证明。')
    else:
        row.update(decision_reason='保留展开4为独立tuner候选：全38历史winner GM+0.2808%，34正median，0个5/5 loser；原9010供4个negative median形状回选。',
                   decision='retain_optional_9011_unroll4',
                   performance={k:full9010[k]for k in ['completed_shapes','geomean_speedup','positive_medians','negative_medians','stable_losers','negative_rows']},
                   caveat='不保证所有形状更快；不把有限等shape GM当作生产负载收益。')
    new_candidates.append(row)

unroll_rows=[]
for name,epoch in [('small9000','small9000_unroll4_split_candidate'),('large9000','large9000_unroll4_split_candidate')]:
    clean('split9000_unroll4_claim.jsonl',epoch)
    unroll_rows+=event_rows(OUT/'loop_unroll4'/f'{name}_event.json')
assert len(unroll_rows)==129
rejected={'status':'reject_9000_unroll4_not_added_to_registry',
          'completed_shapes':129,'geomean_speedup':math.exp(statistics.mean(math.log(r['speedup'])for r in unroll_rows)),
          'positive_medians':sum(r['speedup']>1 for r in unroll_rows),'stable_losers':sum(r['faster_rounds']==0 for r in unroll_rows),
          'rows':unroll_rows,'independent_seed_repeat':event_rows(OUT/'split_formal_corrected/confirm9000_event.json'),
          'reason':'全129明显偏慢；两原稳定正样本新seed仅约0.05%/0.11%且3/5轮，淘汰，候选预算给scale reset。'}
write(OUT/'loop_unroll4/full9000_split_analysis.json',rejected)

work=Path(identity['source_root'])
scope=[]
tracked=[]
for prefix in ['aiter','csrc']:
    tracked+=subprocess.check_output(['git','ls-files',prefix],cwd=ROOT,text=True).splitlines()
for rel in tracked:
    a,b=ROOT/rel,work/rel
    if a.is_file()and b.is_file()and a.read_bytes()!=b.read_bytes():scope.append(rel)
expected={
 'csrc/opus_gemm/opus_gemm_common.py',
 'csrc/opus_gemm/codegen/gen_instances_gfx950.py',
 'csrc/opus_gemm/include/gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_gfx950.cuh',
 'csrc/opus_gemm/include/gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_4wave_256x256_padded_m_gfx950.cuh',
 'csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_4wave_gfx950.cuh',
 'csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_4wave_256x256_padded_m_gfx950.cuh'}
assert set(scope)==expected
source_changes=[{'path':rel,'preapply_sha256':sha(ROOT/rel),'applied_sha256':sha(work/rel)}for rel in scope]
protected={rel:sha(ROOT/rel)for rel in tracked if rel not in scope and (ROOT/rel).is_file()}
for rel in scope:(ROOT/rel).write_bytes((work/rel).read_bytes())
assert all(sha(ROOT/rel)==digest for rel,digest in protected.items())
assert all(not (ROOT/rel).is_file()or not (work/rel).is_file()or (ROOT/rel).read_bytes()==(work/rel).read_bytes()for rel in tracked)
subprocess.run(['git','diff','--check'],cwd=ROOT,check=True)
manifest={**identity,'status':'applied_verified_not_committed','generated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'new_candidates':new_candidates,'source_application':source_changes,
          'all_tracked_aiter_csrc_inputs_equal_formal_worktree':True,
          'production_existing_9000_9010_machine_entries_preserved':True,
          'prior_accepted_902x_and_small_aliases_preserved':True,
          'formal_API':{'path':str(HERE/'api_results.json'),'sha256':sha(HERE/'api_results.json'),'targets':14,'numerical_calls':112,
                        'new9001_targets':6,'new9011_targets':6,'original_controls':2,'owner':api_owner,'performance_timed':False},
          'registry_audit':{'path':str(HERE/'registry_audit.json'),'sha256':sha(HERE/'registry_audit.json')},
          'rejected_9000_unroll4':str(OUT/'loop_unroll4/full9000_split_analysis.json'),
          'supersedes_intermediate_9010_replacement':str(OUT/'formal_selected/integration_manifest.json'),
          'retune_required_for_saved_results_to_choose_new_IDs':True,
          'saved_tuning_csv_changed':False,'pending_experiments':0,'commit_or_push':False,
          'limits':['9001 only an optional finite experimental alternative with small, noisy positive evidence.',
                    'No hard-coded shape dispatch or K specialization added.',
                    'Existing saved9000/9010 ID results use their original device entries; new IDs participate in future tuning.',
                    'New candidates lack their own post-change ATT classification; original evidence guides mechanism only.']}
write(HERE/'integration_manifest.json',manifest)
print(json.dumps({'status':manifest['status'],'new_kids':[r['kid']for r in new_candidates],
                  'original_entries_unchanged':identity['unchanged_existing_entries'],'source_files':len(scope),
                  'official_module':identity['official_module'],'API_calls':112}))
