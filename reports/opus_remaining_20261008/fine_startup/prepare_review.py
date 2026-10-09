#!/usr/bin/env python3
"""Seal exact ISA mechanism and finite numerical/Event targets; CPU only."""
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def instructions(side, name):
    text = (HERE / side / 'kid9062.s').read_text().split('<' + name + '>:', 1)[1].split('\n\n', 1)[0]
    rows = []
    for line in text.splitlines():
        match = re.match(r'^\s*(.*?)\s*// ([0-9a-fA-F]+):\s*(.*?)$', line)
        if match:
            rows.append({'pc': hex(int(match[2], 16)), 'instruction': match[1].strip(), 'words': match[3].strip()})
    return rows


def main():
    audit = json.loads((HERE / 'device_audit.json').read_text())
    assert audit['status'] == 'passed_exact_CPU_build_and_scope_audit'
    entry = next(c for c in audit['checks'] if c['selected'])
    name = audit['selected_symbol']
    sides = {}
    for side in ['baseline', 'candidate']:
        rows = instructions(side, name)
        counts = Counter(r['instruction'].split()[0] for r in rows)
        before_mfma = rows[:next(i for i, r in enumerate(rows) if r['instruction'].startswith('v_mfma_scale'))]
        first_matrix = next(i for i, r in enumerate(before_mfma) if r['instruction'].startswith('buffer_load') and ' lds' in r['instruction'])
        first_scale = next(i for i, r in enumerate(before_mfma) if r['instruction'].startswith('buffer_load') and ' lds' not in r['instruction'])
        first_vmcnt = next(i for i, r in enumerate(before_mfma) if 'vmcnt(' in r['instruction'])
        sides[side] = {'FUNC': entry[side], 'all_static_opcode_counts': dict(counts),
                       'static_request_counts': {k: v for k, v in counts.items() if k.startswith(('buffer_load', 'ds_write', 'buffer_store', 'v_mfma_scale'))},
                       'first_initial_matrix': before_mfma[first_matrix],
                       'first_scale': before_mfma[first_scale],
                       'first_VMEM_wait': before_mfma[first_vmcnt],
                       'matrix_static_requests_before_first_scale_wait': sum(r['instruction'].startswith('buffer_load') and ' lds' in r['instruction'] for r in before_mfma[:first_vmcnt]),
                       'all_vmcnt_texts': [re.search(r'vmcnt\(\d+\)', r['instruction'])[0] for r in rows if 'vmcnt(' in r['instruction']],
                       'all_wait_barrier_rows': [r for r in rows if r['instruction'].startswith(('s_waitcnt', 's_barrier'))],
                       'prologue_request_wait_publish_rows': [r for r in before_mfma if r['instruction'].startswith(('buffer_load', 'ds_write', 's_waitcnt', 's_barrier'))],
                       'disassembly': {'path': str(HERE / side / 'kid9062.s'), 'sha256': sha(HERE / side / 'kid9062.s')}}
    assert sides['baseline']['static_request_counts'] == sides['candidate']['static_request_counts']
    assert sides['baseline']['all_vmcnt_texts'] == sides['candidate']['all_vmcnt_texts']
    assert sides['baseline']['matrix_static_requests_before_first_scale_wait'] == 0
    assert sides['candidate']['matrix_static_requests_before_first_scale_wait'] == 21
    assert sides['baseline']['all_static_opcode_counts']['s_barrier'] == sides['candidate']['all_static_opcode_counts']['s_barrier'] == 5
    assert sides['candidate']['all_static_opcode_counts']['s_waitcnt'] == sides['baseline']['all_static_opcode_counts']['s_waitcnt'] + 1
    report = {'status': 'passed_CPU_exact_scope_ISA_review_pending_GPU', 'cpu_only': True,
              'mechanism': 'Only fixed fine9062 initial matrix/scale issue order',
              'sides': sides, 'request_and_vmcnt_sequences_preserved': True,
              'initial_tile_count': 3, 'ring_stage_count': 4,
              'new_generated_LGKM_wait': {'pc': '0x6604', 'instruction': 's_waitcnt lgkmcnt(0)',
                                        'scope': 'Before last initial B tile requests; no new hardware barrier or new source runtime wait.'},
              'same_selected_resources': {'VGPR': 86, 'SGPR': 70, 'AGPR': 0, 'spill': 0},
              'all_39_unselected_device_entries_exact_equal': True,
              'evidence': {'audit': {'path': str(HERE / 'device_audit.json'), 'sha256': sha(HERE / 'device_audit.json')},
                           'ATT': {'path': str(HERE.parent / 'small_att_analysis.json'), 'sha256': sha(HERE.parent / 'small_att_analysis.json')}},
              'safety_argument': 'The same initial slots are written exactly once before first use. A/B matrix storage and SFA/SFB storage are disjoint. The unchanged scale consumers force their VMEM values ready before LDS publish. All matrix consumers, original wait/barrier sites, ring slot retirement, FP32 partial stores and matching reducer remain unchanged in source. Exact unselected machine code is checked across all18 formal small-family TUs.',
              'performance_uncertainty': 'Scale vmcnt(0) now includes preceding matrix requests. Early full-drain or the extra generated lgkmcnt(0) can lose performance. ATT decoder clocks diagnose one CU; signed reference/repeat/guards and sharedpool5ABBA complete-call Event decide adoption.'}
    (HERE / 'isa_review.json').write_text(json.dumps(report, indent=2) + '\n')
    targets = [(144,7168,16384,'first actual winner, M80 tail and startup-barrier representative'),
               (160,7168,16384,'second actual winner, full M80 tiles'),
               (1,128,16384,'selected fixed type, minimum positive M tail'),
               (79,128,16384,'selected fixed type, M80 minus1 tail'),
               (80,128,16384,'selected fixed type, one full M80 tile'),
               (81,128,16384,'unchanged M96 public-parent branch control'),
               (80,128,128,'unchanged runtime split2 path and empty/one-tile partition control'),
               (80,128,1152,'unchanged runtime split2 uneven4/5 partition ring/drain control')]
    plan = {'name': 'fine9062_initial_handoff_finite_screen', 'baseline': 'Oct8 exact selected baseline, independently rebuilt formal TUs',
            'workspace': True, 'libraries': {side: str(HERE / side / 'experiments.so') for side in ['baseline', 'candidate']},
            'sealed_libraries': {side: {'path': str(HERE / side / 'experiments.so'), 'sha256': sha(HERE / side / 'experiments.so')} for side in ['baseline', 'candidate']},
            'selected_symbol': name, 'single_mechanism': report['mechanism'],
            'device_audit': {'path': str(HERE / 'device_audit.json'), 'sha256': sha(HERE / 'device_audit.json')},
            'targets': [{'kid': 9062, 'shape': [m,n,k], 'seed': 17, 'signed': True, 'split': 2, 'purpose': why,
                         'selected_candidate_type': k == 16384 and not ((m+95)//96 < (m+79)//80)} for m,n,k,why in targets],
            'decision_contract': 'Finite screen only. Two exact current fixed9062 winners must receive same clean physical-card sharedpool5ABBA full producer+reducer Event confirmation before adoption.'}
    (HERE / 'screen_plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    core = dict(plan, name='fine9062_initial_handoff_winner_Event', targets=plan['targets'][:2])
    (HERE / 'winner_event_plan.json').write_text(json.dumps(core, indent=2) + '\n')
    md = '''9062 的单机制候选已准备，尚未进行 GPU 验证。只匹配 `fine_traits<80,1,4,4,1,2,4,128,0,16384>`，把原来的三个初始矩阵 tile 放到原 scale 生产者路径之前，保留四槽 ring。原 SFA/SFB helper、字节、尾行保护、publish、所有消费者/退休/barrier、split2 partial 与 matching reducer 均保留。

基线 ATT 的四波中，慢波首 barrier 到达比最早波晚2308 shader clocks；它执行第二个 SFA pass，包含约1600 clocks 的尾行/对齐保护和打包路径，再串行等待 SFA 与 SFB，导致矩阵请求开始更晚。这是有限单CU的发布到达差，不构成整体 GPU 百分比结论。

CPU构建使用18个原正式 small-family device TU 和同一编译参数，40个device entry基线均精确匹配 Oct8 FUNC/fullmetadata/normalized descriptor；候选只改 fixed9062 producer，39个未选 entry 和所有reducers保持逐字节身份。选中 VGPR86/SGPR70/AGPR0不变，零spill，FUNC由11052B变为10996B。两侧静态global/LDS/store/MFMA opcode计数和全部vmcnt阈值序列一致，硬件barrier均5个。

ISA证明21个静态初始matrix请求位置移到首scale vmcnt之前；这些位置包含每波A的有条件分支，实际请求仍依原wave guard。编译器额外生成一个 `lgkmcnt(0)`，并且原scale `vmcnt(0)`现在包含此前matrix请求，会提前full-drain。这是需要Event解决的性能风险，不能宣称提前请求已产生有效重叠。

`screen_plan.json`包括两项实际赢家、M1/M79/M80尾行与M81未选分支控制、runtime split2短K/不平衡分区控制。`winner_event_plan.json`只含两项固定9062实际赢家。GPU由root统一运行，调用库均不在timed launch中分配workspace，完整调用包含producer与reducer。采用需signed reference/repeat/guards和同物理卡共享池5ABBA完整Event；若失败或退步，保留原selected。
'''
    (HERE / 'review.md').write_text(md)
    print(json.dumps({'status': report['status'], 'screen_targets': len(plan['targets']), 'winner_targets': 2}))


if __name__ == '__main__':
    main()
