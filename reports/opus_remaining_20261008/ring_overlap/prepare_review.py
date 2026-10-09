#!/usr/bin/env python3
"""Seal finite exact ring issue/read-order ISA and root GPU plan."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def ref(path):return {'path':str(path),'sha256':sha(path)}


def main():
    audit=json.loads((HERE/'device_audit.json').read_text()); assert audit['status']=='passed_exact_CPU_build_and_scope_audit'
    checks=[]
    for check in audit['checks']:
        if not check['selected']:continue
        sides={}
        for side in ['baseline','candidate']:
            asm=HERE/side/f"kid{check['parent_id']}.s"
            text=asm.read_text().split('<'+check['name']+'>:',1)[1].split('\n\n',1)[0]
            rows=[]
            for line in text.splitlines():
                match=re.match(r'^\s*(.*?)\s*// ([0-9A-Fa-f]+):',line)
                if match:rows.append({'pc':hex(int(match[2],16)),'instruction':match[1].strip()})
            counts=Counter(r['instruction'].split()[0] for r in rows)
            windows=[]
            for i,r in enumerate(rows):
                if r['instruction']!='s_barrier':continue
                future=rows[i+1:]
                first_mfma=next((n for n,x in enumerate(future) if x['instruction'].startswith('v_mfma_scale')),None)
                if first_mfma is None:continue
                window=future[:first_mfma+1]
                request_positions=[n for n,x in enumerate(window) if x['instruction'].startswith('buffer_load') and ' lds' in x['instruction']]
                read_positions=[n for n,x in enumerate(window) if x['instruction'].startswith('ds_read')]
                if not request_positions:continue
                assert read_positions
                windows.append({'barrier':r,'instructions_through_first_MFMA':window,
                    'future_matrix_issue_before_first_operand_read':max(request_positions)<min(read_positions),
                    'operand_reads_before_future_matrix_issue':max(read_positions)<min(request_positions)})
            sides[side]={'FUNC':check[side],'static_opcode_counts':dict(counts),
                'request_read_write_MFMA_counts':{k:v for k,v in counts.items() if k.startswith(('buffer_load','buffer_store','ds_read','ds_write','v_mfma_scale'))},
                'VMEM_wait_threshold_sequence':[re.search(r'vmcnt\(\d+\)',r['instruction'])[0] for r in rows if 'vmcnt(' in r['instruction']],
                'barrier_to_first_MFMA_ring_windows':windows,'disassembly':ref(asm)}
        assert sides['baseline']['request_read_write_MFMA_counts']==sides['candidate']['request_read_write_MFMA_counts']
        assert sides['baseline']['VMEM_wait_threshold_sequence']==sides['candidate']['VMEM_wait_threshold_sequence']
        assert sides['baseline']['static_opcode_counts']['s_barrier']==sides['candidate']['static_opcode_counts']['s_barrier']==4
        assert all(w['operand_reads_before_future_matrix_issue'] for w in sides['baseline']['barrier_to_first_MFMA_ring_windows'])
        assert all(w['future_matrix_issue_before_first_operand_read'] for w in sides['candidate']['barrier_to_first_MFMA_ring_windows'])
        assert sides['candidate']['static_opcode_counts']['s_waitcnt']==sides['baseline']['static_opcode_counts']['s_waitcnt']+12
        checks.append({'parent_id':check['parent_id'],'symbol':check['name'],'sides':sides})
    assert len(checks)==2
    report={'status':'passed_exact_ring_scope_ISA_pending_GPU','cpu_only':True,'adopted':False,
        'mechanism':'Exact9046/9055 ring future matrix requests precede current operand LDS reads below unchanged group barrier',
        'checks':checks,'all38unselected_entry_identities_equal':True,
        'safety':'Initial scales/matrix prologue, source runtime wait/barrier sequence, per-step futurekt/distance/slot and guards, matrix/scale request byte extents, MFMA accumulation order and drain/output source are unchanged. Reordering moves a future tile only into slots already protected by the original cluster publication/retirement protocol. No runtime barrier moves or per-wave threshold changes.',
        'ISA_effect':'Future matrix request calls now precede current LDS reads. Register allocation becomes smaller and compiler interleaves B fragments/A fragments with MFMA, producing12 extra static waitcnt instructions on each selected kernel. Request/read/write/MFMA opcode counts and vmcnt thresholds are equal. Resource reduction is not evidence of Event gain.',
        'limits':'ATT selected baseline oneCU is local mechanism; compiler scheduling changed as a consequence of one source reorder. Complete-call sharedpool5ABBA Event and signed repeat/guards decide each finite family independently.',
        'evidence':{'device_audit':ref(HERE/'device_audit.json'),'ATT':ref(HERE.parent/'small_att_analysis.json')},
        'script':ref(Path(__file__))}
    (HERE/'isa_review.json').write_text(json.dumps(report,indent=2)+'\n')
    targets=[(9046,[256,7168,16384],'actual winner and baselineATT longK ring representative'),
             (9055,[64,7168,7168],'actual winner and CU1ATT deep-ring representative'),
             (9046,[256,7168,1024],'actual winner, eighttiles with ring partial-cluster drain'),
             (9055,[64,7168,3072],'actual winner,24tile S12C4 ring and drain'),
             (9046,[80,16384,1536],'actual Mtail winner,12tile S6C2 ring'),
             (9055,[48,7168,7168],'actual Mtail winner'),
             (9046,[1,128,128],'shortK single-stage no-ring control and minimum M'),
             (9055,[1,128,128],'shortK single-stage no-ring control and minimum M'),
             (9046,[65,128,896],'sevenKtiles first-ring/partial cluster tail'),
             (9055,[33,128,1664],'thirteenKtiles first-ring/partial cluster tail')]
    plan={'name':'ring_overlap_finite_screen','libraries':{side:str(HERE/side/'experiments.so') for side in ['baseline','candidate']},
          'sealed_libraries':{side:ref(HERE/side/'experiments.so') for side in ['baseline','candidate']},
          'workspace':False,'baseline':'ExactOct8 selected official formal TU rebuild',
          'selected_symbols':[c['symbol'] for c in checks],'single_mechanism':report['mechanism'],
          'device_audit':ref(HERE/'device_audit.json'),'ISA_review':ref(HERE/'isa_review.json'),
          'targets':[{'kid':kid,'shape':shape,'seed':17,'signed':True,'purpose':why} for kid,shape,why in targets],
          'decision_contract':'Screen numerical8repeat and finiteEvent; each selected family requires clean same-card sharedpool5ABBA Event on its own representative. Broaden winner coverage only after representative positive evidence.'}
    (HERE/'screen_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    core=dict(plan,name='ring_overlap_representative_complete_Event',targets=plan['targets'][:2])
    (HERE/'representative_event_plan.json').write_text(json.dumps(core,indent=2)+'\n')
    (HERE/'review.md').write_text('''9046/9055 的共享small LDS ring单机制候选已准备，尚未GPU验证。只匹配当前9046 S6/C2和9055 S12/C4原traits，保留原symbol。每个ring step把原未来matrix prefetch从当前operand LDS reads之后放到之前；原cluster wait/barrier仍在两者之前，future slot/index/kt/guard保持相同。初始scale/matrix handoff、scales、短K、drain、输出均保留。

9046 baselineATT的普通tile boundary median316 clocks、cluster boundary648；cluster后首LDS operand read常见84/168clocks，原未来matrix issue在10个matrix LDS reads及scale读后。9055 CU1显示S12/C4 cluster boundary median492、group内216；同样未来matrix issue在current operands之后。候选检验这一可交换请求次序对后续tile供给是否有益，不改已被拒绝的fine_wait阈值或N48/N128几何。

CPU构建18个正式small-family TU，40entry基线逐项精确Oct8 FUNC/fullmetadata/descriptor；候选仅两producer变化，38个其它entry完全不变。9046VGPR80→62/SGPR55保持/FUNC8044→7972；9055VGPR56→48/SGPR73→72/FUNC10356→10204，零spill。两侧VMEM/LDS读写/MFMA opcode计数及vmcnt阈值序列一致，s_barrier各4。ISA证明各cluster后未来matrix issue在operand read前；编译器因此将fragment读交错MFMA，静态waitcnt各多12条。这些resource/调度差异需要完整Event判断，不构成收益证据。

screen_plan包含两个ATT实际赢家、K1024/K3072、Mtail实际赢家、短K无ring控制和首ring partial-cluster尾部。representative_event_plan仅两个有限ATT赢家，以同物理卡共享pool5ABBA完整调用Event分别决策。若代表退步，该机制在对应family关闭；若正收益，才扩大该family实际winner范围并完成正式API检查后采用。
''')
    print(json.dumps({'status':report['status'],'screen_targets':len(targets),'representative_targets':2}))


if __name__=='__main__':main()
