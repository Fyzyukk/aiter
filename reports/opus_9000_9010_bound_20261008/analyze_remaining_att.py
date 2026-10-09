from pathlib import Path
import json,statistics,collections,importlib.util,hashlib
ROOT=Path.cwd();OUT=ROOT/'reports/opus_9000_9010_bound_20261008';DIAG=OUT/'remaining_bound_diagnostics'
sp=importlib.util.spec_from_file_location('elf',ROOT/'reports/opus_remaining_20261008/formal_selected/audit_integration.py');elf=importlib.util.module_from_spec(sp);sp.loader.exec_module(elf)
plan=json.loads((DIAG/'plan.json').read_text());official={r['name']:r for b in elf.bundles(plan['official_binary']) for r in b['rows']};records=[];pending=[]
owner_audit=json.loads((OUT/'owner_audit.json').read_text())
owner_record=next(r for r in owner_audit['records'] if Path(r['path']).resolve()==(DIAG/'claim.jsonl').resolve())
assert not owner_record['incomplete']
owner_epochs={e['name']:e for e in owner_record['epochs']}
assert owner_epochs['additional_config_correctness']['strict_clean']
def stats(v):return {'count':len(v),'median':statistics.median(v),'min':min(v),'max':max(v)} if v else None
for idx,target in enumerate(plan['targets']):
 folder=DIAG/f'target{idx}_kid{target["kid"]}_att';app=folder/'application.json';uis=list(folder.glob('ui_output_*'))
 if not app.exists() or not uis:pending.append(idx);continue
 a=json.loads(app.read_text());d=json.loads((uis[0]/'code.json').read_text())
 if a['status']!='passed' or not d['code']:pending.append(idx);continue
 epoch=owner_epochs[f'target{idx}_kid{target["kid"]}_att'];assert epoch['strict_clean']
 assert a['rows'][0]['actual_official_module']['sha256']==plan['official_binary_sha256']
 code={r[2]:r for r in d['code']};objects={r[4] for r in d['code']};assert len(objects)==1
 co=next(folder.glob(f'*code_object_id_{next(iter(objects))}.out'));meta=elf.COMMON.summarize_image(co.read_bytes());kernel=next(r for r in meta if r['name']==target['symbol'])
 for key in ['instruction_sha256','metadata','descriptor_normalized_sha256']:assert kernel[key]==official[kernel['name']][key]
 waves=[]
 def op(line):return code[line][0]
 for path in uis[0].glob('se*_sm*_sl*_wv*.json'):
  doc=json.loads(path.read_text());w=doc.get('wave',{});events=w.get('instructions',[]);mfma=[e for e in events if op(e[4]).startswith('v_mfma_scale')]
  if not mfma:continue
  expected=target['expected_mfma_total']//(target['producer_workgroups']*target['waves'])
  actual=target['actual_ids'][0]
  if actual==9054:
   allowed={8,16}
   if len(mfma) not in allowed:continue
  elif len(mfma)!=expected:continue
  first=mfma[0][0]+mfma[0][2];last=mfma[-1][0]+mfma[-1][2];begin=w['begin'];end=w['end'];cats=collections.Counter();main=collections.Counter();startup=collections.Counter();deps=collections.defaultdict(set)
  for line,items in w.get('waitcnt',[]):
   for item in items:deps[line].add(op(item[0]))
  for j,e in enumerate(events):
   instr=op(e[4]);category='other';members=deps[e[4]]
   if instr.startswith('v_mfma_scale'):category='MFMA_issue_or_dependency'
   elif instr=='s_barrier':category='barrier'
   elif instr.startswith('s_waitcnt'):
    if any(x.startswith('s_load') for x in members) and not any(x.startswith(('ds_','buffer_')) for x in members):category='SMEM_dependency'
    elif any(x.startswith('buffer_load') for x in members):category='VMEM_dependency'
    elif any(x.startswith('ds_') for x in members):category='LDS_dependency'
    else:category='unresolved_wait'
   elif instr.startswith('buffer_load'):category='VMEM_issue'
   elif instr.startswith(('buffer_store','global_store')):category='output_VMEM_issue'
   elif instr.startswith('ds_'):category='LDS_issue'
   elif instr.startswith('s_load'):category='SMEM_issue'
   extent=max(0,min(e[0]+e[3],events[j+1][0] if j+1<len(events) else end)-e[0]);cats[category]+=extent
   main[category]+=max(0,min(e[0]+extent,last)-max(e[0],first));startup[category]+=max(0,min(e[0]+extent,first)-max(e[0],begin))
  waves.append({'file':path.name,'life':end-begin,'startup':first-begin,'after_last_mfma':end-last,'MFMA_count':len(mfma),'mfma_stall_clocks':sum(e[2] for e in mfma),'mfma_issue_clocks':sum(e[3]-e[2] for e in mfma),'classes':dict(cats),'main_classes':dict(main),'startup_classes':dict(startup)})
 if not waves:pending.append(idx);continue
 if target['actual_ids'][0]==9054:
  # K=768 has six K128 tiles over four wave-K partitions.
  # Two partitions execute one tile and two execute two tiles.
  # Eight MFMAs per tile give a sorted multiset of 8/8/16/16.
  counts=collections.Counter(w['MFMA_count'] for w in waves)
  assert counts=={8:2,16:2} and sum(w['MFMA_count'] for w in waves)==48
  count_validation={'expected_counts_per_complete_workgroup_sorted':[8,8,16,16],
                    'observed_counts_sorted':sorted(w['MFMA_count'] for w in waves),
                    'expected_workgroup_sum':48,'observed_workgroup_sum':48,
                    'note':'Uneven K partitions; 12 is an arithmetic average, not a per-wave expectation.'}
 else:
  count_validation={'expected_per_wave':expected,'observed_counts':sorted(set(w['MFMA_count'] for w in waves))}
 summary={'waves':len(waves),'expected_MFMA_per_wave':expected,'life':stats([w['life']for w in waves]),'startup':stats([w['startup']for w in waves]),'after_last_mfma':stats([w['after_last_mfma']for w in waves]),'classes':{k:statistics.median(w['classes'].get(k,0)for w in waves)for k in set().union(*(w['classes']for w in waves))},'main_classes':{k:statistics.median(w['main_classes'].get(k,0)for w in waves)for k in set().union(*(w['main_classes']for w in waves))},'startup_classes':{k:statistics.median(w['startup_classes'].get(k,0)for w in waves)for k in set().union(*(w['startup_classes']for w in waves))}}
 if target['actual_ids'][0]==9054:summary['expected_MFMA_per_wave']=[8,8,16,16]
 selected_target=dict(target)
 selected_target['historical_reported_traits']=target['traits']
 demangled=kernel['demangled'];prefix=target['kernel_function']+'<'
 selected_target['traits']=demangled.split(prefix,1)[1].rsplit('>(opus_',1)[0].strip()
 record={'index':idx,'target':selected_target,'capture':str(folder),'kernel':kernel,'summary':summary,'waves':waves,
         'MFMA_count_validation':count_validation,'strict_owner':{'claim_path':owner_record['path'],
         'claim_sha256':owner_record['sha256'],'command_name':epoch['name'],'owner':epoch['owner'],
         'GPU':epoch['gpu'],'monitor_count':epoch['monitor_count'],'strict_clean':True}};records.append(record)
 print(json.dumps({'idx':idx,'kid':target['kid'],'actual':target['actual_ids'],'shape':target['shape'],'summary':summary}))
(OUT/'remaining_att_analysis.json').write_text(json.dumps({'status':'passed_finite' if not pending else 'partial','captures':records,'pending':pending,'limits':['Every row is its own exact-config representative, not all-winner classification','Shader-clock shares are local and correlated','Memory bandwidth saturation or frontend dispatch cannot follow from wave wait alone']},indent=2)+'\n')
