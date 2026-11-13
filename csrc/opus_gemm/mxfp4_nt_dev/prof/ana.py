import csv,glob,os,sys,json,collections
sys.path.insert(0,'/root/workspace/355_1_trans/mxfp4_gemm_4wave_20261112/tools/prof')
def cat(op):
    if 'mfma' in op: return 'MFMA'
    for p,n in (('buffer_load','buffer_load'),('buffer_store','buffer_store'),('ds_read','ds_read'),('ds_write','ds_write'),('v_accvgpr','accvgpr'),('s_waitcnt','waitcnt')):
        if op.startswith(p): return n
    if op=='s_barrier': return 'barrier'
    if op=='s_nop': return 'nop'
    if op.startswith('v_'): return 'VALU'
    if op.startswith('s_'): return 'SALU'
    return 'other'
names=sys.argv[1:]; R={}
for d in names:
    f=glob.glob(f'{d}/att/stats_ui_output_agent_*_dispatch_1.csv')[0]
    rows=[r for r in csv.DictReader(open(f)) if r['Instruction'] and not r['Instruction'].startswith(';')]
    c=collections.defaultdict(lambda:[0,0,0])
    for r in rows:
        k=cat(r['Instruction'].split()[0]); c[k][0]+=int(r['Hitcount'] or 0); c[k][1]+=int(r['Latency'] or 0); c[k][2]+=int(r['Stall'] or 0)
    ws=[]
    for g in glob.glob(f'{d}/att/ui_output*/se0_sm*_sl0_wv*.json'):
        x=json.load(open(g))['wave']; ws.append((int(x['begin']),int(x['end'])))
    span=max(e for b,e in ws)-min(b for b,e in ws)
    R[d]=(c,span)
print('| 类别 | '+' | '.join(f'{os.path.basename(n)} 条数/tile · cyc/条 · stall' for n in names)+' |'); print('|---|'+'---|'*len(names))
for k in ['MFMA','ds_read','buffer_load','VALU','SALU','buffer_store','ds_write','waitcnt','barrier','nop']:
    print(f'| {k} | '+' | '.join(f'{R[n][0][k][0]/16:.0f} · {R[n][0][k][1]/max(1,R[n][0][k][0]):.1f} · {R[n][0][k][2]/16:.0f}' for n in names)+' |')
print('| CU 上 4 个 SIMD 的总时长(cycle) | '+' | '.join(f'{R[n][1]}' for n in names)+' |')
