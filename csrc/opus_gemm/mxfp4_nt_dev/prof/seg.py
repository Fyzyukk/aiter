# loop 外（hit == 每 tile 一次）的代码按 s_barrier 切段：每段每 tile 的 cycle、MFMA 数、主要 stall
import json,glob,sys,collections
d=sys.argv[1]; W=int(sys.argv[2]) if len(sys.argv)>2 else 16
c=json.load(open(glob.glob(f'{d}/att/ui_output*/code.json')[0]))['code']
mx=max(r[6] for r in c if r[0] and 'mfma' in r[0])
segs=[]; cur=[]
for r in c:
    if not r[0] or r[6]==0: continue
    op=r[0].split()[0]
    if r[6]==mx:   # main loop
        if cur and cur[-1]!='LOOP': segs.append(cur); cur=[]
        if not segs or segs[-1]!='LOOP': segs.append('LOOP')
        continue
    cur.append(r)
    if op=='s_barrier' or op=='s_endpgm' or op.startswith('s_cbranch'): segs.append(cur); cur=[]
if cur: segs.append(cur)
tot=0
for i,s in enumerate(segs):
    if s=='LOOP': print(f'---- main loop ----'); continue
    lat=sum(r[7] for r in s)/W; tot+=lat; nm=sum(r[6] for r in s if 'mfma' in r[0])/W
    if lat<150: continue
    st=collections.Counter()
    for r in s: st[r[0].split()[0]]+=r[8]/W
    top=', '.join(f'{k}:{v:.0f}' for k,v in st.most_common(4) if v>=20)
    hits=sorted({r[6] for r in s})
    print(f'#{i:3d} hit{hits[:3]} cyc/tile {lat:6.0f}  MFMA {nm:5.0f} (理想 {nm*16:5.0f})  多出 {lat-nm*16:6.0f} | stall: {top} | 结尾: {s[-1][0][:40]}')
print(f'loop 外合计 {tot:.0f} cycle/tile')
