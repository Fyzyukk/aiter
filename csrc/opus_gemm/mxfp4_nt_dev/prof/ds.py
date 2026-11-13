import csv,glob,sys,collections,re
for d in sys.argv[1:]:
    f=glob.glob(f'{d}/att/stats_ui_output_agent_*_dispatch_1.csv')[0]
    rows=[r for r in csv.DictReader(open(f)) if r['Instruction'] and not r['Instruction'].startswith(';')]
    g=collections.defaultdict(lambda:[0,0,0])
    for r in rows:
        ins=r['Instruction']
        if not ins.startswith('ds_read'): continue
        op=ins.split()[0]; off=re.search(r'offset:(\d+)',ins); 
        g[op][0]+=int(r['Hitcount']); g[op][1]+=int(r['Latency']); g[op][2]+=int(r['Stall'])
    print(d, ' | '.join(f'{k}: {v[0]/16:.0f}条 {v[1]/max(1,v[0]):.1f}cyc/条 stall/tile {v[2]/16:.0f}' for k,v in sorted(g.items())))
