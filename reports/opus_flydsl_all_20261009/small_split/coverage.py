#!/usr/bin/env python3
"""Audit shape coverage and emulate partition/ring indices on CPU only."""
import collections,csv,hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
REG={9040,9041,9042,9051,9052,9053,9054}
FINE={9060,9061,9062,9063}
VARIANTS={110:(16,16,1,1,4),111:(16,16,1,2,4),120:(16,32,1,1,4),121:(16,32,1,2,4),
          130:(32,32,1,1,4),140:(32,64,1,1,4),210:(48,64,4,1,1),211:(48,64,4,1,2),
          220:(64,128,4,1,1),221:(64,128,4,1,2),230:(96,128,4,1,1),231:(96,128,4,1,2)}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def partition(total,count,index):
    per,extra=divmod(total,count)
    return index*per+min(index,extra),per+(index<extra)
def ring_register(loops):
    q=3;slots={i:i for i in range(min(q,loops))};seen=[];loads=list(slots.values())
    full,tail=divmod(loops,q)
    for group in range(max(0,full-1)):
        for i in range(q):
            seen.append(slots[i]);slots[i]=(group+1)*q+i;loads.append(slots[i])
    if full:
        for i in range(q):
            seen.append(slots[i])
            if i<tail:slots[i]=full*q+i;loads.append(slots[i])
    for i in range(tail):seen.append(slots[i])
    assert seen==list(range(loops)) and sorted(loads)==list(range(loops))
def ring_fine(loops):
    stages=4;distance=3;initial=loops if loops<=stages else distance
    slots={i:i for i in range(initial)};seen=[];loads=list(slots.values())
    drain=max(0,loops-distance)
    for kt in range(loops):
        slot=kt%stages
        if loops>stages and kt<drain and kt+distance<loops:
            destination=(slot+distance)%stages
            slots[destination]=kt+distance;loads.append(kt+distance)
        assert slots[slot]==kt
        seen.append(slots[slot])
    assert seen==list(range(loops)) and sorted(loads)==list(range(loops))
def main():
    rows=list(csv.DictReader((HERE/'frozen/losers294.csv').open()))
    assigned=[r for r in rows if int(r['opus_parent_kid']) in REG|FINE]
    assert len(assigned)==90 and len({(r['M'],r['N'],r['K']) for r in assigned})==90
    covered=[];candidate_rows=[]
    for r in assigned:
        m,n,k=map(int,(r['M'],r['N'],r['K']));fine=int(r['opus_parent_kid']) in FINE
        ids=[v for v in VARIANTS if (v>=200)==fine]
        covered.append({**{p:r[p] for p in ['M','N','K','opus_parent_kid','actual_kid','actual_tile_M','actual_tile_N','actual_waves','global_splitK','fixed_K']},
            'pool':'fine' if fine else 'register','candidate_ids':';'.join(map(str,ids)),
            'baseline_workspace_bytes':int(r['global_splitK'])*m*n*4 if int(r['global_splitK'])>1 else 0})
        for v in ids:
            bm,bn,mn,local,split=VARIANTS[v]
            assert m<=(2048 if fine else 512) and n%128==0 and k%128==0 and k<=16384 and n%bn==0
            alltiles=[]
            for s in range(split):
                start,loops=partition(k//128,split,s)
                for wk in range(local):
                    begin,count=partition(loops,local,wk)
                    tiles=list(range(start+begin,start+begin+count));alltiles.extend(tiles)
                    (ring_fine if fine else ring_register)(count)
                    # Native A/SFA and preshuffled B/SFB remain absolute K128.
                    for kt in tiles:
                        assert 0<=kt<k//128
                        assert kt*m+(m-1)<m*(k//128)
                        assert (n//128-1)*(k//128)+kt<(n//128)*(k//128)
            assert sorted(alltiles)==list(range(k//128))
            workspace=split*m*n*4 if split>1 else 0
            if fine:
                loops=(k//128+split-1)//split;active=min(4,loops)
                lds=active*((bm+bn)//8*1056)+(bm+1)*loops
            else:lds=(local-1)*bm*bn*4
            assert lds<=160*1024
            candidate_rows.append({'M':m,'N':n,'K':k,'parent':r['opus_parent_kid'],'actual':r['actual_kid'],'variant':v,
                'tile_M':bm,'tile_N':bn,'tile_K':128,'wave_M':(1 if bm==48 or not fine else 2),
                'wave_N':(4 if bm==48 and fine else 2 if fine else 1),'wave_K':local,'workgroup_waves':mn*local,
                'global_splitK':split,'workspace_bytes':workspace,'kernel_calls':2 if split>1 else 1,
                'producer_grid_x':n//bn,'producer_grid_y':(m+bm-1)//bm,'producer_grid_z':split,
                'reducer_grid_x':(m*n+2047)//2048 if split>1 else 0,'producer_lds_bytes':lds,
                'scale_K_group':128,'numerical_validation':'pending_gpu_stop','performance_validation':'pending_gpu_stop'})
    # Include every legal K128 count, including empty SK4 partitions/local waves.
    for total in range(1,129):
        for split in (1,2,4):
            for local in (1,2):
                seen=[]
                for s in range(split):
                    start,loops=partition(total,split,s)
                    for wk in range(local):
                        begin,count=partition(loops,local,wk)
                        seen.extend(range(start+begin,start+begin+count));ring_register(count)
                        ring_fine(count)
                assert sorted(seen)==list(range(total))
    # Reducer Vec16 mapping covers logical C once; all N are multiples of 128.
    for r in assigned:
        elems=int(r['M'])*int(r['N']);assert elems%16==0
        for index in range(0,elems,16):assert index+15<elems
    for name,data in [('coverage.csv',covered),('candidate_cases.csv',candidate_rows)]:
        with (HERE/name).open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    report={'status':'cpu_partition_coverage_passed','cpu_only':True,'gpu_operations':0,'assigned_unique_shapes':len(assigned),
            'register_shapes':sum(int(r['opus_parent_kid']) in REG for r in assigned),
            'fine_shapes':sum(int(r['opus_parent_kid']) in FINE for r in assigned),'candidate_cases':len(candidate_rows),
            'parent_counts':dict(sorted(collections.Counter(r['opus_parent_kid'] for r in assigned).items())),
            'candidate_case_counts':dict(sorted(collections.Counter(r['variant'] for r in candidate_rows).items())),
            'partition_checks':'all K128 totals 1..128, global SK1/2/4, local WK1/2; exact once including empty partitions',
            'register_queue':'P3 loads/computes exactly once; absolute K scale groups',
            'fine_queue':'S4/C1 read-only drain loads/computes exactly once; absolute K scale groups',
            'reducer':'Vec16/Block128 logical ranges exactly once, N multiple 128',
            'limitations':'CPU index/contract/layout checks do not establish GPU numerical correctness, synchronization behavior, or performance',
            'source_sha256':sha(HERE/'coverage.py'),'coverage_sha256':sha(HERE/'coverage.csv'),'candidate_cases_sha256':sha(HERE/'candidate_cases.csv')}
    (HERE/'coverage_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
