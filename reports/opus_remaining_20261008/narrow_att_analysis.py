#!/usr/bin/env python3
"""Analyze current corrected narrow ATT captures using CPU reads only.

Writes only this new script's adjacent analysis JSON/Markdown. Original decoder
raw durations remain unchanged; exclusive interval partitions cap overlaps.
"""
import collections
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import statistics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
D = HERE / 'diagnostics'

def load(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def norm(value): return re.sub(r'\s+', ' ', value.strip())
def require(value, message):
    if not value: raise ValueError(message)
def describe(values):
    values = sorted(values)
    return {'samples': len(values), 'min': min(values), 'median': statistics.median(values), 'mean': statistics.mean(values), 'p90': values[int(.9 * (len(values) - 1))], 'max': max(values)} if values else {'samples': 0}

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod

COMMON = module('narrow_common', ROOT / 'reports/opus_resume_20261008/formal_selected/smoke_common.py')
MERGED = module('narrow_claim', D / 'merged_att_analysis.py')

def basic_kind(op):
    if op.startswith('s_load'): return 'kernarg_SMEM_issue'
    if op.startswith('buffer_load_dwordx4'): return 'matrix_directLDS_VMEM_issue' if ' lds' in op else 'SFA_vector16_VMEM_issue'
    if op.startswith('buffer_load_ubyte'): return 'scale_byte_VMEM_issue'
    if op.startswith('ds_read_u16'): return 'SFA_u16_LDS_read'
    if op.startswith('ds_read_u8'): return 'SFB_u8_LDS_read'
    if op.startswith('ds_read'): return 'matrix_LDS_read'
    if op.startswith('ds_write'): return 'scale_LDS_publish'
    if op.startswith('v_mfma_scale'): return 'scaled_MFMA'
    if op.startswith('v_readlane'): return 'scalar_lane_restore'
    if op.startswith('v_writelane'): return 'scalar_lane_save'
    if op.startswith('v_perm'): return 'SFA_pack_or_default_permute'
    if op.startswith('s_nop'): return 'explicit_nop'
    if op.startswith('buffer_store'): return 'C_VMEM_store'
    if op.startswith('v_cvt_pk_bf16'): return 'BF16_conversion'
    if op == 's_endpgm': return 'endpgm_decoder_duration'
    if op == 's_barrier': return 'barrier'
    if op.startswith('s_waitcnt'): return 'waitcnt'
    return 'other_instructions'

def event(e, code, base, kinds, origin=0):
    r = code[e[4]]
    return {'code_line': e[4], 'relative_pc': hex(r[5] - base), 'pc': hex(r[5]), 'instruction': r[0], 'kind': kinds[e[4]], 'time': e[0] - origin, 'stall': e[2], 'duration': e[3], 'successful_issue': e[0] + e[2] - origin}

def partition(events, kinds, start, end):
    require(end >= start, 'Negative interval')
    result = collections.Counter()
    for i, e in enumerate(events):
        eend = min(e[0] + e[3], events[i+1][0]) if i+1 < len(events) else e[0] + e[3]
        overlap = max(0, min(end, eend) - max(start, e[0]))
        result[kinds[e[4]]] += overlap
    require(sum(result.values()) <= end-start, 'Interval partition double counts')
    result['unattributed_decoder_gap'] += end-start-sum(result.values())
    return dict(sorted((k,v) for k,v in result.items() if v))

def capture(index, target, plan, queue, claims, references, linked):
    name = f'target{index}_kid{target["kid"]}_att'
    folder = D / name
    app = load(folder / 'application.json')
    require(app['status'] == 'passed' and app['profiling_only'] and len(app['rows']) == 1, 'Incomplete app')
    r = app['rows'][0]
    require(r['target_index'] == index and r['kid'] == target['kid'] and r['shape'] == target['shape'], 'Target mismatch')
    require(app['official_binary_sha256'] == plan['official_binary_sha256'] == r['actual_official_module']['sha256'], 'Module mismatch')
    command = next(c for c in queue['commands'] if c['name'] == name)
    claim = MERGED.clean_claim(name, app, command, queue, claims)
    ui = list(folder.glob('ui_output_agent_*'))
    require(len(ui) == 1, 'Missing single decoded UI')
    ui = ui[0]
    raw = load(ui / 'code.json')
    require(raw['header'] == 'ISA, _, LineNumber, Source, Codeobj, Vaddr, Hit, Latency, Stall, Idle', 'Decoder schema mismatch')
    code = {r[2]: r for r in raw['code']}
    require(len(code) == len(raw['code']), 'Duplicate code lines')
    oid = {r[4] for r in code.values()}
    require(len(oid) == 1, 'Multiple code objects')
    image = next(folder.glob(f'*_code_object_id_{next(iter(oid))}.out'))
    require(sha(image) in linked, 'Captured image not in current linked audit')
    elf = COMMON.Elf(image.read_bytes())
    ref = references[target['symbol']]
    metadata = next(k for k in elf.metadata()['amdhsa.kernels'] if k['.name'] == target['symbol'])
    symbol = elf.symbols()[target['symbol']]
    descriptor = bytearray(elf.symbols()[metadata['.symbol']]['bytes']); descriptor[16:24] = b'\0'*8
    require(symbol['size'] == ref['instruction_bytes'] and hashlib.sha256(symbol['bytes']).hexdigest() == ref['instruction_sha256'] == target['instruction_sha256'], 'Captured FUNC differs')
    require(metadata == ref['metadata'] and hashlib.sha256(descriptor).hexdigest() == ref['descriptor_normalized_sha256'], 'Captured metadata/descriptor differs')
    base, size = MERGED.symbol_address(elf, target['symbol'])
    label = next(r for r in code.values() if r[0].startswith(';'))
    require(target['symbol'] in label[0] and label[5] == base, 'Decoder symbol/base differs')
    historical = (ROOT / ref['retained_disassembly']).read_text()
    header = next(m for m in re.finditer(r'^([0-9a-f]+) <([^>]+)>:', historical, re.M) if m[2] == target['symbol'])
    old_base = int(header[1],16)
    body = historical[header.end():]; end = re.search(r'^([0-9a-f]+) <',body,re.M)
    if end: body = body[:end.start()]
    isa = {int(m[2],16)-old_base:norm(m[1]) for m in re.finditer(r'^\s*(.*?)\s*//\s*([0-9A-F]+):',body,re.M)}
    require(len(isa) == len(code)-1, 'Incomplete decoder ISA function')
    for row in code.values():
        if row[0].startswith(';'): continue
        require(base <= row[5] < base+size and norm(row[0]) == isa[row[5]-base], 'Decoder PC ISA mismatch')
    csv_rows = list(csv.DictReader(next(folder.glob('stats_ui_output*.csv')).open()))
    require(len(csv_rows) == len(raw['code']), 'CSV coverage differs')
    for a,b in zip(csv_rows,raw['code']):
        require(a['Instruction'] == b[0] and int(a['CodeObj']) == b[4] and int(a['Vaddr']) == b[5] and [int(a[x]) for x in ('Hitcount','Latency','Stall','Idle')] == b[6:10], 'CSV mismatch')
    paths = sorted(ui.glob('se*_sm*_sl*_wv*.json'))
    objects = [load(p) for p in paths]
    waves = [o['wave'] for o in objects]
    unit = 8 if target['kid'] == 9023 else 4
    expected = target['shape'][2]//128*unit
    require(waves and len(waves)%4 == 0, 'Incomplete wave groups')
    aggregates = collections.defaultdict(lambda:[0,0,0]); deps = collections.defaultdict(set)
    for o,w in zip(objects,waves):
        x=w['instructions']
        require(o['num_insts'] == o['num_stitched'] == len(x) and o['duration'] == w['end']-w['begin'], 'Partial stitched wave')
        require(w['cu'] == 0 and all(len(e) == 5 and e[3] >= e[2] >= 0 for e in x), 'Wave contract mismatch')
        require(all(a[0]+a[2] <= b[0] for a,b in zip(x,x[1:])), 'Issue order differs')
        require(sum('v_mfma_scale' in code[e[4]][0] for e in x) == expected, 'Dynamic MFMA differs')
        require(code[x[-1][4]][0] == 's_endpgm', 'Missing wave end')
        for e in x:
            v=aggregates[e[4]];v[0]+=1;v[1]+=e[3];v[2]+=e[2]
        for line,members in w['waitcnt']:
            deps[line].update(m[0] for m in members)
    require(all(aggregates[r[2]] == r[6:9] for r in code.values()), 'Wave/stat/code aggregate mismatch')
    kinds = {line:basic_kind(r[0]) for line,r in code.items()}
    for line,members in deps.items():
        member_kinds = {basic_kind(code[m][0]) for m in members}
        if basic_kind(code[line][0]) != 'waitcnt': continue
        kinds[line] = 'wait_' + '+'.join(sorted(member_kinds))
    bypc = {r[5]-base:r[2] for r in code.values() if not r[0].startswith(';')}
    refill_bars = {0x2a94,0x3b0c,0x4b88,0x6024} if target['kid'] == 9023 else set()
    panel_bar = 0x1420 if target['kid'] == 9023 else 0x14dc
    first_smem = 0x20 if target['kid'] == 9023 else 0x18
    records=[]; gaps=collections.defaultdict(list); breakdowns=collections.defaultdict(list); examples=collections.defaultdict(list); role_events=collections.defaultdict(list); all_readlane=collections.Counter()
    for path,w in zip(paths,waves):
        x=w['instructions']; pos=[i for i,e in enumerate(x) if 'v_mfma_scale' in code[e[4]][0]]; issues=[x[i][0]+x[i][2] for i in pos]
        get=lambda pc:next(e for e in x if code[e[4]][5]-base == pc)
        smem=get(first_smem);bar=get(panel_bar)
        bounds=[w['begin'],smem[0]+smem[3],bar[0]+bar[3],issues[0]]
        labels=['first_kernarg_chain','remaining_startup_matrix_scale_publication','operand_reads_and_runtime_setup']
        prologue=[{'phase':label,'clocks':b-a,'breakdown':partition(x,kinds,a,b)} for label,a,b in zip(labels,bounds,bounds[1:])]
        raw_counts=collections.Counter(code[e[4]][0].split()[0] for e in x)
        readlane=[event(e,code,base,kinds,w['begin']) for e in x if code[e[4]][0].startswith('v_readlane')]
        all_readlane.update(e['relative_pc'] for e in readlane)
        refills=[]
        for j,e in enumerate(x):
            pc=code[e[4]][5]-base
            if pc in refill_bars:
                previous=sum(p<j for p in pos)
                refills.append({'barrier':event(e,code,base,kinds,w['begin']),'preceding_MFMA_count':previous,'preceding_tile':(previous-1)//unit,'next_MFMA_ordinal':previous})
            if kinds[e[4]] != 'other_instructions' and (e[0] < issues[0] or pc in refill_bars or code[e[4]][0].startswith('v_readlane')):
                role_events[hex(pc)].append({'wave':path.name,**event(e,code,base,kinds,w['begin'])})
        gap_records=[]
        for ordinal,(p,q) in enumerate(zip(pos,pos[1:])):
            tile,ix=divmod(ordinal,unit);between=x[p+1:q];refill=any(code[e[4]][5]-base in refill_bars for e in between)
            phase='startup' if tile==0 else ('drain' if tile >= target['shape'][2]//128-3 else 'interior')
            label=f'slot{w["slot"]}/{phase}/gap{ix}/'+('refill' if refill else 'ordinary')
            value=issues[ordinal+1]-issues[ordinal];part=partition(x[p:q+1],kinds,issues[ordinal],issues[ordinal+1]);gaps[label].append(value);breakdowns[label].append(part)
            if phase=='interior' and (ix==0 or refill or ix==unit-1): examples[label].append((value,{'wave':path.name,'K128_tile':tile,'gap_index':ix,'gap_clocks':value,'breakdown':part,'events':[event(e,code,base,kinds,issues[ordinal]) for e in x[p:q+1]]}))
            if refill:gap_records.append({'tile':tile,'gap_index':ix,'clocks':value,'breakdown':part})
        stores=[e for e in x if code[e[4]][0].startswith('buffer_store')]
        overlap=[{'relative_pc':hex(code[a[4]][5]-base),'next_relative_pc':hex(code[b[4]][5]-base),'clocks':a[0]+a[3]-b[0]} for a,b in zip(x,x[1:]) if a[0]+a[3]>b[0]]
        records.append({'wave':path.name,'wave_sha256':sha(path),'cu':w['cu'],'simd':w['simd'],'slot':w['slot'],'decoder_wave_id':w['id'],'WG_id_exported':False,'complete':True,'begin':w['begin'],'end':w['end'],'duration':w['end']-w['begin'],'dynamic_MFMA':len(pos),'prologue_to_first_MFMA':issues[0]-w['begin'],'first_to_last_MFMA':issues[-1]-issues[0],'last_MFMA_to_wave_end':w['end']-issues[-1],'prologue_share_percent':100*(issues[0]-w['begin'])/(w['end']-w['begin']),'first_store_from_wave_begin':stores[0][0]+stores[0][2]-w['begin'],'first_dynamic_MFMA':event(x[pos[0]],code,base,kinds,w['begin']),'prologue_phases':prologue,'whole_wave_partition':partition(x,kinds,w['begin'],w['end']),'dynamic_counts':dict(raw_counts),'readlane_events':readlane,'scale_vector16_events':[event(e,code,base,kinds,w['begin']) for e in x if basic_kind(code[e[4]][0])=='SFA_vector16_VMEM_issue'],'byte_VMEM_events':[event(e,code,base,kinds,w['begin']) for e in x if code[e[4]][0].startswith('buffer_load_ubyte')],'refill_publish_barriers':refills,'refill_MFMA_gaps':gap_records,'initial_barrier_release':bar[0]+bar[3],'decoder_duration_overlaps':overlap,'timeline_minus_wave_duration':sum(t[1] for t in w['timeline'])-(w['end']-w['begin']),'event_end_minus_wave_end':x[-1][0]+x[-1][3]-w['end'],'prologue_events':[event(e,code,base,kinds,w['begin']) for e in x if e[0]<issues[0] and kinds[e[4]] != 'other_instructions']})
    groups=[]
    for r in sorted(records,key=lambda x:x['initial_barrier_release']):
        t=r['initial_barrier_release']
        if not groups or t-groups[-1]['release_min']>4: groups.append({'release_min':t,'release_max':t,'waves':[]})
        groups[-1]['release_max']=t;groups[-1]['waves'].append(r['wave'])
    require(all(len(g['waves'])==4 for g in groups),'Publication groups not full4wave')
    dependency_records=[{'relative_pc':hex(code[line][5]-base),'instruction':code[line][0],'member_instructions':[{'relative_pc':hex(code[m][5]-base),'instruction':code[m][0],'kind':basic_kind(code[m][0])} for m in sorted(members)]} for line,members in sorted(deps.items())]
    summary={key:describe([r[key] for r in records]) for key in ('duration','prologue_to_first_MFMA','first_to_last_MFMA','last_MFMA_to_wave_end','prologue_share_percent')}
    windows={key:{'gap':describe(vals),'breakdown_mean_clocks':{kind:statistics.mean(p.get(kind,0) for p in breakdowns[key]) for kind in sorted(set().union(*breakdowns[key]))}} for key,vals in sorted(gaps.items())}
    representatives={key:min(vals,key=lambda x:abs(x[0]-statistics.median(gaps[key])))[1] for key,vals in examples.items()}
    role_summaries=[{'relative_pc':pc,'instruction':values[0]['instruction'],'count':len(values),'duration':describe([v['duration'] for v in values]),'duration_sum':sum(v['duration'] for v in values),'events':values} for pc,values in sorted(role_events.items(),key=lambda x:int(x[0],16))]
    return {'target_index':index,'parent_id':target['kid'],'shape':target['shape'],'status':'passed','symbol':target['symbol'],'traits':target['traits'],'identity':{'official_module_sha256':plan['official_binary_sha256'],'captured_code_object':str(image),'captured_code_object_sha256':sha(image),'captured_FUNC_sha256':ref['instruction_sha256'],'captured_FUNC_bytes':size,'captured_FUNC_vaddr':hex(base),'normalized_descriptor_sha256':ref['descriptor_normalized_sha256'],'metadata_equal_current':True,'all_relative_PC_instructions_equal_current':True,'matched_instruction_count':len(isa),'CSV_code_equal':True,'wave_aggregate_equal_code':True,'app_sha256':sha(folder/'application.json'),'code_json_sha256':sha(ui/'code.json')},'clean_claim':claim,'wave_count':len(records),'complete_waves':len(records),'expected_MFMA_per_wave':expected,'MFMA_per_K128_tile':unit,'publication_release_groups':groups,'publication_group_note':'Synchronized release infers local WG cohorts; SIMD/slot alone is not a WG ID. EXEC masks and true WG IDs are not exported.','wave_summary':summary,'per_slot_summary':{str(slot):{key:describe([r[key] for r in records if r['slot']==slot]) for key in ('duration','prologue_to_first_MFMA')} for slot in sorted({r['slot'] for r in records})},'dynamic_readlane_relative_PC_counts':dict(all_readlane),'wait_dependency_sets':dependency_records,'role_events_by_PC':role_summaries,'MFMA_issue_gap_groups':windows,'representative_gaps':representatives,'waves':records,'limits':['Instruction events do not reveal EXEC active lanes or byte traffic.','This finite capture covers CU0; M544 tail WG and fixed grouped-M tail are not established.','Correlated waves and gap repetitions are not independent performance repeats.','Raw event durations can overlap4clocks; exclusive partitions cap to next event and wave endpoint while preserving aggregate identity.']}

def claim_exclusion(index, target, claims, reason):
    """Document an observed unowned process without relaxing clean_claim."""
    require(reason == 'Monitor contains a process other than the recorded owner',
            'Only an evidenced unowned-process claim rejection can be excluded')
    name = f'target{index}_kid{target["kid"]}_att'
    app_path = D / name / 'application.json'
    app = load(app_path)
    start = max((c for c in claims if c.get('event') == 'start'
                 and c.get('command', {}).get('name') == name
                 and c['time'] <= app['started']), key=lambda c:c['time'])
    end = min((c for c in claims if c.get('event') == 'end'
               and c.get('name') == name and c['time'] >= app['finished']),
              key=lambda c:c['time'])
    epoch = [c for c in claims if start['time'] <= c['time'] <= end['time']]
    owners = [c for c in epoch if c.get('event') == 'owner_identity'
              and c.get('name') == name]
    require(len(owners) == 1, 'Exclusion needs exactly one recorded owner')
    owner = owners[0]
    monitors = [c for c in epoch if c.get('event') == 'monitor'
                and c['time'] >= owner['time']]
    offending = [c for c in monitors
                 if any(p['pid'] != owner['host_pid'] for p in c['processes'])]
    require(offending, 'Exclusion needs an observed unowned process')
    return {'target_index':index, 'parent_id':target['kid'], 'shape':target['shape'],
            'status':'excluded_unowned_GPU_process', 'strict_clean_claim_rejection':reason,
            'command_name':name, 'command_epoch':[start['time'],end['time']],
            'owner_identity':owner, 'offending_monitors':offending,
            'unowned_host_pids':sorted({p['pid'] for c in offending for p in c['processes']
                                       if p['pid'] != owner['host_pid']}),
            'command_end_record':end, 'app_path':str(app_path), 'app_sha256':sha(app_path),
            'analysis_used':False, 'timing_comparison_used':False,
            'note':'The observed monitor overrides the end record contamination:false. '
                   'Original capture files are retained; this analysis does not decode or use '
                   'the rejected capture. A clean rerun is required for a paired comparison.'}

def main():
    plan=load(D/'plan.json');queue=load(D/'att_queue.json');review=load(HERE/'narrow_review.json');audit=load(ROOT/'reports/opus_resume_20261008/formal_selected/identity_audit.json')
    require(sha(plan['official_binary'])==plan['official_binary_sha256']==review['baseline']['sha256'],'Official module changed')
    references={v['symbol']:v for v in review['variants']};linked={c['candidate_image_sha256'] for c in audit['linked_module_bundle_checks']}
    claims=[json.loads(line) for line in (D/'att_claim_corrected.jsonl').read_text().splitlines()]
    captures=[];errors=[];exclusions=[]
    for index,target in enumerate(plan['targets']):
        if target['kid'] not in (9023,9024):continue
        try:captures.append(capture(index,target,plan,queue,claims,references,linked))
        except (ValueError,KeyError,StopIteration,FileNotFoundError) as error:
            if str(error) == 'Monitor contains a process other than the recorded owner':
                exclusions.append(claim_exclusion(index,target,claims,str(error)))
            else:errors.append({'target_index':index,'error':str(error)})
    findings=['Runtime9023 each of eight complete waves executes46 scalar lane saves but only6 lane restores: three remainder control and three output address restores. None of the captured ordinary-tile or aligned refill paths executes the static120/readlane branch body. Unroll1 is not justified by a steady restore bottleneck.','Runtime9023 refills occur after MFMA ordinals248/504/760 (tile31/63/95 before final eight MFMA of that tile). The7→0 gaps containing producer wait/pack/publish/barrier are far wider than ordinary7→0 gaps. Full-vector SFA waits precede SFBissue and its wait; this is a concrete local serialization candidate.','Accepted fixed9024 target11 contains224 MFMA per wave and zero lane operations, with no panel refills. Startup median is5218 shaderclocks, about15% of its33644-clock median wave life. Two wave timelines execute SFA vectors, one executes SFB; source preserves matching guards and byte/u16 layout.','Both runtime9023 captures show two synchronized4wave publication cohorts in slots0/1. Slot1 life is longer in these captures; the exporter does not identify true WG coordinates and this does not establish whole-device dispatch fill.','Fixed9024 252/264WG targets use the same fixed/group4 entry in the source/ELF review. Target12 is excluded for an observed unowned GPU process; these captures do not provide a clean252/264WG timing comparison or evidence of a256 source boundary.']
    report={'status':'failed' if errors else ('passed_with_exclusion' if exclusions else 'passed'),'generated_utc':datetime.now(timezone.utc).isoformat(),'cpu_only':True,'new_GPU_execution':False,'new_build_execution':False,'kernel_modified':False,'analysis_script_sha256':sha(__file__),'plan_sha256':sha(D/'plan.json'),'queue_sha256':sha(D/'att_queue.json'),'claim_sha256_snapshot':sha(D/'att_claim_corrected.jsonl'),'review_sha256':sha(HERE/'narrow_review.json'),'timing_contract':'Shaderclocks; successfulissue=time+stall, duration includesstall. Wait queue members are not individualmemory-returntimestamps. Exclusive partitions cap decoder overlap; rawstataggregates preserved.','captures':captures,'exclusions':exclusions,'errors':errors,'findings':findings,'adoption_status':'No candidate adoption; isolated issue-before-publish candidate requires exact baseline/ISA, guard/numerical and shared-address AB/BA Event.'}
    (HERE/'narrow_att_analysis.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    lines=['# Current Oct8 narrow ATT analysis','',report['timing_contract'],'','| Target / shape | Complete waves | MFMA each | Prologue median | Wave life median | Readlane each |','| --- | --- | --- | --- | --- | --- |']
    for c in captures:
        r=c['wave_summary'];lines.append(f"| {c['target_index']} / {'×'.join(map(str,c['shape']))} | {c['complete_waves']} | {c['expected_MFMA_per_wave']} | {r['prologue_to_first_MFMA']['median']} | {r['duration']['median']} | {len(c['waves'][0]['readlane_events'])} |")
    lines+=['','All times are shader clocks from finite correlated CU0 waves. For accepted targets9/10/11, exact current module/CO/FUNC/metadata/descriptor, relative PC ISA, corrected clean claim, complete MFMA count and code/CSV/wave aggregates passed.','']
    for e in exclusions:
        lines += [f"Target{e['target_index']} is excluded: {e['strict_clean_claim_rejection']}. During epoch{e['command_epoch'][0]}..{e['command_epoch'][1]}, owner hostPID{e['owner_identity']['host_pid']} and unowned hostPID(s){e['unowned_host_pids']} appear in the same GPU monitor. The end record says contamination:false, but the strict owner check rejects the capture. Original evidence is retained and no timing or decoded-wave finding from this capture is used.", '']
    lines+=['- '+x for x in findings]
    lines+=['','Full queue sets, dynamic PCs, phase partitions, producer events and ordinary/refill/drain gap records are in [narrow_att_analysis.json](narrow_att_analysis.json). Original durations are retained;4clock overlapping decoder scalar/nop events are capped only for exclusive time partitions. Tail coordinates and EXEC masks are not exported, so no tail-WG or active-byte claim follows.','',report['adoption_status'],'']
    (HERE/'narrow_att_analysis.md').write_text('\n'.join(lines))
    print(json.dumps({'status':report['status'],'accepted':[c['target_index'] for c in captures],'excluded':[c['target_index'] for c in exclusions],'errors':errors}))
    if errors:raise SystemExit(1)

if __name__=='__main__':main()
