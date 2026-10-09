#!/usr/bin/env python3
"""Seal exact9051 ISA request order and finite screen/Event plan."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return {'path':str(p),'sha256':sha(p)}


def main():
    audit=json.loads((HERE/'device_audit.json').read_text());assert audit['status']=='passed_exact_CPU_build_and_scope_audit'
    check=next(c for c in audit['checks'] if c['selected']);sides={}
    for side in ['baseline','candidate']:
        asm=HERE/side/'kid9051.s';text=asm.read_text().split('<'+check['name']+'>:',1)[1].split('\n\n',1)[0];rows=[]
        for line in text.splitlines():
            match=re.match(r'^\s*(.*?)\s*// ([0-9A-Fa-f]+):',line)
            if match:rows.append({'pc':hex(int(match[2],16)),'instruction':match[1].strip()})
        counts=Counter(r['instruction'].split()[0] for r in rows)
        prologue=rows[:next(i for i,r in enumerate(rows) if r['instruction'].startswith('v_mfma_scale'))]
        requests=[r for r in prologue if r['instruction'].startswith('buffer_load')]
        pattern=['scale_byte' if r['instruction'].startswith('buffer_load_ubyte') else 'matrix16B' for r in requests]
        assert len(pattern)==27
        expected=(['matrix16B']*2+['scale_byte']+['matrix16B']*4+['scale_byte']*2) if side=='baseline' else ['scale_byte']*3+['matrix16B']*6
        assert pattern==expected*3
        sides[side]={'FUNC':check[side],'static_opcode_counts':dict(counts),
                     'request_compute_LDS_store_counts':{k:v for k,v in counts.items() if k.startswith(('buffer_load','buffer_store','v_mfma_scale','ds_read','ds_write'))},
                     'initial3tile_request_pattern':pattern,'initial_request_instructions':requests,
                     'B_matrix_cache_sc0_nt_count':sum('sc0 nt' in r['instruction'] and r['instruction'].startswith('buffer_load_dwordx4') for r in rows),
                     'all_waits':[r for r in rows if r['instruction'].startswith('s_waitcnt')],
                     'first_operand_VMEM_wait':next(r for r in prologue if 'vmcnt(' in r['instruction']),
                     'disassembly':ref(asm)}
    assert sides['baseline']['request_compute_LDS_store_counts']==sides['candidate']['request_compute_LDS_store_counts']
    assert sides['baseline']['B_matrix_cache_sc0_nt_count']==sides['candidate']['B_matrix_cache_sc0_nt_count']
    assert sides['baseline']['static_opcode_counts']['s_barrier']==sides['candidate']['static_opcode_counts']['s_barrier']==1
    report={'status':'passed_exact9051_issue_order_ISA_pending_GPU','cpu_only':True,'adopted':False,
            'mechanism':'Exactruntime9051 eachtile SFA/SFB byte requests before A/B matrix requests',
            'sides':sides,'other39deviceentries_exactequal':True,
            'scope':'Runtime9051<16,32,1,1,3,4,3,3,0,false,false> only; originalQ3, waveK4, noBscaleReuse, cache3, perfragment byte/tail guards, queue retirement, FP32reduce and packedBF16output.',
            'safety':'Scale values are assigned to the same queue fields with identical guards. Matrix slices/cache controls use original offsets. No queue field is consumed earlier; compute, advance, full/tail loops, waveK LDS partial retirement/barrier/reduction/output source are unchanged. Only compiler sched_barrier(0) is added between issue groups, no hardware wait/barrier.',
            'ISA_effect':'Initialtile request order is now3scale-byte then6matrix16B requests, from matrix2/scale1/matrix4/scale2. Total24byte/48matrix requests,16MFMA,2LDSwrites,6LDSreads and1barrier equal. Compiler wait counts/thresholds adjust to reordered operands (firstvmcnt1→2), preserving source synchronization. VGPR104→102/SGPR55→52 does not establish gain.',
            'limits':'Historicalexact9051 LFIFO/translation/cache counter evidence and localATT issue/wait stalls diagnose request-pressure paths. Event must decide; longdecoderissue stalls are not individual HBM latency. 32ATTwaves acrossmultipleWG prevent intraWGbarrier inference.',
            'evidence':{'device_audit':ref(HERE/'device_audit.json'),'ATT':ref(HERE.parent/'small_att_analysis.json')},'script':ref(Path(__file__))}
    (HERE/'isa_review.json').write_text(json.dumps(report,indent=2)+'\n')
    cases=[([1,65536,1536],'actual winner and exactLFIFO/ATT external representative'),
           ([16,7168,3072],'actual winner longerqueue refill and fullM'),
           ([1,7168,768],'actual shortK winner, unequalwaveK2/1 tile partitions'),
           ([2,7168,16384],'actual longK/Mtail winner'),
           ([1,128,128],'minimumM/K, threeemptywaveK partitions'),
           ([15,128,896],'Mtail, uneven7tile waveK partition'),
           ([17,128,1152],'crossM16tile tail, uneven9tile partitions'),
           ([16,128,1536],'fullM/onefullQ3group perwave'),
           ([16,128,1664],'fullM/Q3refillanduneven13tilepartition')]
    plan={'name':'runtime9051_scale_first_finite_screen','workspace':False,
          'baseline':'Oct8 exactselected formalTU rebuild','libraries':{side:str(HERE/side/'experiments.so') for side in ['baseline','candidate']},
          'sealed_libraries':{side:ref(HERE/side/'experiments.so') for side in ['baseline','candidate']},
          'selected_symbol':check['name'],'single_mechanism':report['mechanism'],
          'device_audit':ref(HERE/'device_audit.json'),'ISA_review':ref(HERE/'isa_review.json'),
          'targets':[{'kid':9051,'shape':shape,'seed':17,'signed':True,'purpose':why} for shape,why in cases],
          'decision_contract':'Finite numerical/repeat/guards and two clean samecard sharedpool5ABBA completeEvent representatives. Only after bothpositivescreen should all17actualwinner coverage be measured for adoption.'}
    (HERE/'screen_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    event=dict(plan,name='runtime9051_scale_first_representative_Event',targets=plan['targets'][:2])
    (HERE/'representative_event_plan.json').write_text(json.dumps(event,indent=2)+'\n')
    (HERE/'review.md').write_text('''runtime9051 的scale-first请求次序候选已准备，尚未GPU验证。仅匹配原 `<16,32,1,1,Q3,waveK4,OUTPUT3,cache3,K0,false,false>` type，保留原symbol和false B-scale reuse。每tile先发原SFA/SFB byte请求，再发原A/B matrix请求；字节/数量/guard/地址/cache/queue生命周期/compute/FP32归约及输出source保留，只用compiler sched_barrier锁定两个issue组次序。

基线ATT的32完整wave跨多个WG，startupmatrix issue和scaleissue/wait占主要局部时段，精确身份与原LFIFO/translation/cache四组counter可复用。候选检验byte请求提前是否改善same-tile请求供给，不改变整体requests或采用已拒全局reuse。长issue decoderstall不等同单次HBM访问延迟；跨WG barrier跨度不用于WG内arrival结论。

18正式TU/40entry基线均精确Oct8 FUNC/fullmetadata/descriptor；候选39个未选entry完全相同。选中FUNC3012→3000B/VGPR104→102/SGPR55→52，零spill。ISA证明每initialtile由matrix2/SFA1/matrix4/SFB2变为scale3/matrix6，三tile27requests覆盖一致；总matrix48/byte24、MFMA16/LDSwrites2/reads6/barrier1及B sc0nt请求数相同。Compiler operandwait随请求次序重新匹配，首vmcnt1→2、staticwait22→21，这是需要数值/完整Event判定的调度变化。

screen_plan覆盖external/longqueue/shortK实际赢家以及Mtail/emptywave/unevenpartition/Q3refill。representative_event_plan仅external[1,65536,1536]和longqueue[16,7168,3072]两个实际赢家，使用同物理卡51共享池5轮AB/BA完整call Event；若代表无正收益就关闭，只有正证据才扩展17个实际winner覆盖并走正式API检查。
''')
    print(json.dumps({'status':report['status'],'screen_targets':len(cases),'representative_Event_targets':2}))


if __name__=='__main__':main()
