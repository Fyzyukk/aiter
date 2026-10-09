from pathlib import Path
import json,statistics,hashlib,collections,importlib.util
ROOT=Path.cwd();OUT=ROOT/'reports/opus_9000_9010_bound_20261008';DIAG=OUT/'baseline_diagnostics'
spec=importlib.util.spec_from_file_location('elf',ROOT/'reports/opus_remaining_20261008/formal_selected/audit_integration.py');elf=importlib.util.module_from_spec(spec);spec.loader.exec_module(elf)
plan=json.loads((DIAG/'plan.json').read_text());captures=[]
def stats(v):return {'count':len(v),'median':statistics.median(v),'min':min(v),'max':max(v)} if v else None
for index,target in enumerate(plan['targets'][:5]):
 folders=list(DIAG.glob(f'target{index}_kid{target["kid"]}_att*'));valid=[]
 for folder in folders:
  ui=list(folder.glob('ui_output_*'));app=folder/'application.json'
  if not ui or not app.exists():continue
  d=json.loads((ui[0]/'code.json').read_text())
  if not d['code']:continue
  a=json.loads(app.read_text());assert a['status']=='passed';assert a['rows'][0]['actual_official_module']['sha256']==plan['official_binary_sha256']
  valid.append((folder,ui[0],d,a))
 if not valid:continue
 folder,ui,d,app=valid[0];code={r[2]:r for r in d['code']};objects={r[4] for r in d['code']};assert len(objects)==1
 image=next(folder.glob(f'*code_object_id_{next(iter(objects))}.out'));_,rows=elf.COMMON.summarize_image(image.read_bytes()) if isinstance(elf.COMMON.summarize_image(image.read_bytes()),tuple) else (None,elf.COMMON.summarize_image(image.read_bytes()))
 kernel=next(r for r in rows if r['name']==target['symbol']);assert kernel['instruction_sha256']==target['instruction_sha256']
 waves=[]
 for path in ui.glob('se*_sm*_sl*_wv*.json'):
  doc=json.loads(path.read_text());w=doc.get('wave',doc)
  if 'instructions' not in w:continue
  events=w['instructions'];mfma=[e for e in events if code[e[4]][0].startswith('v_mfma_scale')]
  if len(mfma)!=target['shape'][2]//128*64:continue
  begin=events[0][0];end=events[-1][0]+events[-1][3];first=mfma[0][0]+mfma[0][2];last=mfma[-1][0]+mfma[-1][2]
  waits=[];cats=collections.Counter();issues=collections.Counter()
  for j,e in enumerate(events):
   op=code[e[4]][0];extent=min(e[0]+e[3],events[j+1][0] if j+1<len(events) else end)-e[0];extent=max(0,extent)
   cat='MFMA' if op.startswith('v_mfma_scale') else 'barrier' if op=='s_barrier' else 'wait' if op.startswith('s_waitcnt') else 'LDS' if op.startswith('ds_') else 'VMEM' if op.startswith('buffer_') else 'SMEM' if op.startswith('s_load') else 'other'
   cats[cat]+=extent;issues[cat]+=1
   if cat in ['barrier','wait']:
    waits.append({'pc':hex(code[e[4]][5]),'op':op,'clocks':e[3],'before_first_mfma':e[0]<first,'in_main':first<=e[0]<last})
  waves.append({'path':path.name,'life':end-begin,'first_mfma':first-begin,'after_last_mfma':end-last,'MFMA_count':len(mfma),'clocks_by_event_class':dict(cats),'issue_counts':dict(issues),'long_waits':sorted(waits,key=lambda r:r['clocks'],reverse=True)[:12]})
 if not waves:continue
 record={'index':index,'kid':target['kid'],'shape':target['shape'],'capture':str(folder),'kernel':kernel,'waves':waves,'summary':{'wave_count':len(waves),'life':stats([w['life'] for w in waves]),'startup':stats([w['first_mfma'] for w in waves]),'after_last_mfma':stats([w['after_last_mfma'] for w in waves]),'clocks_by_event_class_median':{k:statistics.median(w['clocks_by_event_class'].get(k,0) for w in waves) for k in set().union(*(w['clocks_by_event_class'] for w in waves))}}}
 captures.append(record);print(json.dumps({k:record[k] for k in ['index','kid','shape','summary']}))
(OUT/'att_analysis.json').write_text(json.dumps({'status':'partial' if len(captures)<5 else 'passed_finite','captures':captures,'limits':['CU-local shader clocks, not full-kernel time','Wait duration is not individual request latency','Complete waves and instruction identity checked; owner log audit remains separate']},indent=2)+'\n')
