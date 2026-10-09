from pathlib import Path
import json,statistics,math,csv
OUT=Path(__file__).resolve().parent
results=[]
for p in sorted(OUT.glob('*/screen_event.json'))+sorted(OUT.glob('*/full*_event.json')):
 d=json.loads(p.read_text());rows=[]
 for r in d['rows']:
  e=r.get('event_confirmation',{})
  if e.get('status')!='passed':continue
  by={(m['round'],m['label']):m for m in e['measurements']}
  pairs=[by[(i,'baseline')]['us_per_call']/by[(i,'candidate')]['us_per_call'] for i in range(5)]
  rows.append({'kid':r['kid'],'shape':r['shape'],'speedup':e['median_speedup']['candidate'],'faster_rounds':sum(v>1 for v in pairs),'paired_speedups':pairs,'median_us':e['median_us']})
 summary={'path':str(p),'status':d['status'],'completed_shapes':len(rows),'geomean_speedup':math.exp(statistics.mean(math.log(r['speedup']) for r in rows)) if rows else None,'positive_medians':sum(r['speedup']>1 for r in rows),'stable_losers':sum(r['faster_rounds']==0 for r in rows),'rows':rows}
 results.append(summary)
 print(json.dumps({k:v for k,v in summary.items() if k!='rows'}))
(OUT/'event_analysis.json').write_text(json.dumps({'status':'computed_from_retained_Event','experiments':results},indent=2)+'\n')
