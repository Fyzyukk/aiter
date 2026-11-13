# main loop 按 quadrant（每 32 条 MFMA 一段）统计每趟 cycle、ds_read/buffer_load/MFMA 的 stall
import json,glob,sys,collections
names=['C10','C01','C11','C00']
for d in sys.argv[1:]:
    c=json.load(open(glob.glob(f'{d}/att/ui_output*/code.json')[0]))['code']
    mx=max(r[6] for r in c if r[0] and "mfma" in r[0]); L=[r for r in c if r[0] and r[6]==mx]
    q=0; nm=0; S=collections.defaultdict(lambda:collections.Counter())
    for r in L:
        op=(r[0] or '').split()[0] if r[0] else ''
        h=max(1,r[6]); k='q%d'%min(nm//32,7)
        S[k]['cyc']+=r[7]/h
        if 'mfma' in op: S[k]['mfma_stall']+=r[8]/h; nm+=1
        elif op.startswith('ds_read'): S[k]['ds_read_n']+=1; S[k]['ds_read_stall']+=r[8]/h
        elif op.startswith('buffer_load'): S[k]['bl_n']+=1; S[k]['bl_stall']+=r[8]/h
        elif op.startswith('ds_write'): S[k]['dsw_n']+=1
        elif op=='s_barrier' or op.startswith('s_waitcnt'): S[k]['wait']+=r[8]/h
    print(f'== {d}  (每趟 2 个 k-tile, 共 {nm} MFMA)')
    print(' quad   cycle  ds_read条/stall  buf_load条/stall  ds_write  MFMA stall  wait/barrier')
    for i in range(8):
        s=S['q%d'%i]; print(f' {"偶" if i<4 else "奇"}{names[i%4]}  {s["cyc"]:6.0f}   {s["ds_read_n"]:3.0f} / {s["ds_read_stall"]:5.0f}     {s["bl_n"]:3.0f} / {s["bl_stall"]:5.0f}      {s["dsw_n"]:3.0f}     {s["mfma_stall"]:5.0f}     {s["wait"]:5.0f}')
