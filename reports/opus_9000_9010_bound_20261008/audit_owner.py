from pathlib import Path
import json,hashlib
OUT=Path(__file__).resolve().parent
records=[]
for path in sorted(OUT.rglob('*claim.jsonl')):
 lines=[json.loads(l) for l in path.read_text().splitlines() if l.strip()];epochs={};current=None;gpu=None
 for e in lines:
  if e['event']=='claimed':gpu=e['gpu']
  elif e['event']=='start':
   current={'name':e['command']['name'],'start':e['time'],'command':e['command'],'fingerprint':e['fingerprint'],'gpu':gpu,'unknown_pids':[],'monitor_count':0};epochs[current['name']]=current
  elif e['event']=='owner_identity' and current:current['owner']=e['host_pid']
  elif e['event']=='monitor' and current:
   current['monitor_count']+=1
   current['unknown_pids']+= [p['pid'] for p in e['processes'] if p['pid']!=current.get('owner')]
  elif e['event']=='end' and current:
   current['end']=e['time'];current['returncode']=e['returncode'];current['contamination']=e['contamination'];current['strict_clean']=not current['unknown_pids'] and not e['contamination'] and e['returncode']==0 and current.get('owner') is not None
   current=None
 records.append({'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'epochs':list(epochs.values()),'incomplete':current is not None})
result={'status':'partial' if any(r['incomplete'] for r in records) else 'passed' if all(e.get('strict_clean') for r in records for e in r['epochs']) else 'failed','records':records}
(OUT/'owner_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'status':result['status'],'completed_epochs':sum('end' in e for r in records for e in r['epochs']),'unclean':[(e['name'],e['unknown_pids']) for r in records for e in r['epochs'] if e.get('strict_clean') is False]}))
