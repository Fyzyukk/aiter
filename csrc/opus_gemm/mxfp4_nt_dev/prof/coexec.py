# Co-execution analysis of the hottest loop: for each MFMA, cycles until the next MFMA issues
# (sum of per-instruction latency, per hit).  Ideal = MFMA_CYC.  Excess is attributed to the
# instructions issued between the two MFMAs (their latency beyond the MFMA's own slot).
import json,sys,collections,glob
MFMA_CYC=int(sys.argv[2]) if len(sys.argv)>2 else 16
c=json.load(open(glob.glob(sys.argv[1]+'/att/ui_output*/code.json')[0]))['code']
mx=max(r[6] for r in c if r[0] and 'mfma' in r[0])
loop=[r for r in c if r[0] and r[6]==mx]
hits=mx
def cls(op):
    if 'mfma' in op: return 'MFMA'
    if op.startswith('ds_read'): return 'ds_read'
    if op.startswith('ds_write'): return 'ds_write'
    if op.startswith('buffer_load'): return 'buffer_load'
    if op.startswith('s_waitcnt'): return 'waitcnt'
    if op=='s_barrier': return 'barrier'
    if op.startswith('s_nop'): return 's_nop'
    if 'permlane' in op: return 'permlane'
    if op.startswith('v_'): return 'VALU'
    if op.startswith('s_'): return 'SALU'
    return 'other'
idx=[i for i,r in enumerate(loop) if 'mfma' in r[0]]
nm=len(idx)
gaps=[]; excess_by=collections.Counter(); mfma_stall=0; pair=[]
for j,i in enumerate(idx):
    nxt=idx[j+1] if j+1<nm else len(loop)
    seg=loop[i:nxt]
    gap=sum(r[7] for r in seg)/hits
    gaps.append(gap)
    mfma_stall+=loop[i][8]/hits
    ex=gap-MFMA_CYC
    if ex>0:
        # attribute: MFMA's own latency beyond MFMA_CYC? (pipe busy / dependency) + others' latency
        own=loop[i][7]/hits
        others=[(cls(r[0].split()[0]), r[7]/hits) for r in seg[1:]]
        tot_o=sum(x for _,x in others)
        # time the MFMA slot covers others: MFMA_CYC - own (if own < MFMA_CYC)
        hidden=max(0,MFMA_CYC-own)
        rem=ex
        if own>MFMA_CYC: excess_by['MFMA 自身停顿(等前一条MFMA/依赖)']+=min(rem,own-MFMA_CYC); rem-=min(rem,own-MFMA_CYC)
        if rem>0 and tot_o>0:
            for k,x in others: excess_by[k]+=rem*x/tot_o
total=sum(gaps)
print(f'main loop: {nm} MFMA per trip, {hits} trips(hits), total {total:.0f} cycles/trip, ideal {nm*MFMA_CYC} -> MFMA pipe busy {nm*MFMA_CYC/total:.1%}')
h=collections.Counter()
for g in gaps:
    b='=16(满)' if g<=MFMA_CYC+0.5 else '17-24' if g<=24 else '25-32' if g<=32 else '33-64' if g<=64 else '>64'
    h[b]+=1
print('MFMA 间隔分布:', dict(sorted(h.items())))
print(f'总的多出时间 {total-nm*MFMA_CYC:.0f} cycles/trip，按原因：')
for k,v in excess_by.most_common(): print(f'   {k:32s} {v:7.0f}  ({v/(total-nm*MFMA_CYC):.0%})')
# pairs (kk0,kk1 on the same C): gap of the 2 MFMAs together
pg=[gaps[j]+gaps[j+1] for j in range(0,nm-1,2)]
hp=collections.Counter('=32(满)' if g<=2*MFMA_CYC+0.5 else '33-40' if g<=40 else '41-64' if g<=64 else '>64' for g in pg)
print('每对 MFMA（同一个 C 的 kk=0,1）总时间分布:', dict(sorted(hp.items())))
print('最差的 10 处（MFMA 后面跟的指令）:')
order=sorted(range(nm), key=lambda j:-gaps[j])[:10]
for j in order:
    i=idx[j]; nxt=idx[j+1] if j+1<nm else len(loop)
    seg=loop[i:nxt]
    print(f'   gap {gaps[j]:6.1f}: '+' | '.join(f"{r[0].split()[0]}({r[7]/hits:.0f})" for r in seg)[:200])
