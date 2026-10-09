"""Build an exact-entry bottleneck inventory from retained, finite evidence."""
from pathlib import Path
import collections, csv, datetime, hashlib, importlib.util, json, statistics

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
OLD = ROOT / 'reports/opus_remaining_20261008'
RESUME = ROOT / 'reports/opus_resume_20261008'
DOC = Path('/root/workspace/trans_github/mi300_gpu_bandwidth_chapters3_4_5_notes.md')

def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def short(p): return str(Path(p).relative_to(OUT)) if Path(p).is_relative_to(OUT) else str(Path(p).relative_to(ROOT))

spec = importlib.util.spec_from_file_location('elf', OLD / 'formal_selected/audit_integration.py')
elf = importlib.util.module_from_spec(spec); spec.loader.exec_module(elf)
previous = read(OLD / 'formal_selected/identity_audit.json')
inventory = {r['symbol']: r for r in read(OLD / 'inventory.json')['entries']}
progress = {r['symbol']: r for r in read(OLD / 'family_progress.json')['entries']}
old_small = {r['symbol']: r for r in read(OLD / 'small_att_analysis.json')['captures']}
merged = collections.defaultdict(list)
for r in read(OLD / 'diagnostics/merged_att_analysis.json')['captures']: merged[r['symbol']].append(r)
narrow = collections.defaultdict(list)
for r in read(OLD / 'narrow_att_analysis.json')['captures']: narrow[r['symbol']].append(r)
additional = {r['target']['symbol']: r for r in read(OUT / 'remaining_att_analysis.json')['captures']}
large = read(OUT / 'att_analysis.json')['captures']
baseline_private = {r['name']: r for r in read(OUT / 'baseline/build_manifest.json')['rows']}
owners = read(OUT / 'owner_audit.json')
owner_by_name = {e['name']: (r, e) for r in owners['records'] for e in r['epochs']}

def record_evidence(path, captures, final_kernel, kernel_source=None):
    evidence = {'path': short(path), 'sha256': sha(path), 'captures': captures}
    if kernel_source is not None:
        evidence['captured_instruction_sha256'] = kernel_source['instruction_sha256']
        evidence['same_machine_identity_as_final'] = all(kernel_source[k] == final_kernel[k]
                for k in ['instruction_sha256', 'metadata', 'descriptor_normalized_sha256'])
    return evidence

rows = []
for parent in previous['parents']:
    kid = parent['parent_id']
    for variant in parent['variants']:
        kernel = variant['candidate']; symbol = variant['name']; inv = inventory.get(symbol)
        traits = inv['traits'] if inv else kernel['demangled'].split('<', 1)[1].rsplit('>(opus_', 1)[0].strip()
        if symbol.startswith('_Z43opus_gemm_mxscale_bpreshuffle_reduce_kernel'):
            traits = kernel['demangled']
        if inv and kid in [9042, 9053, 9054]:
            traits = kernel['demangled'].split('_kernel<', 1)[1].rsplit('>(opus_', 1)[0].strip()
        row = {'parent_id': kid, 'symbol': symbol, 'traits': traits,
               'actual_ids': inv['actual_configuration_ids'] if inv else [kid],
               'historical_winner_count': inv['winner_count'] if inv else {9000:129,9010:38,9021:42}.get(kid,0),
               'role': 'winner_producer' if inv or kid in [9000,9010,9021] else 'support_producer',
               'primary': 'unconfirmed', 'secondary': [], 'confidence': '未确认',
               'classification_scope': '未作本配置独立瓶颈测量', 'representatives': [],
               'finding': '', 'evidence': [], 'kernel_identity': kernel,
               'optimization_state': progress[symbol]['current_optimization_state'] if symbol in progress else 'keep_current',
               'memory_bandwidth_saturation_proven': False, 'dispatch_bound_proven': False}
        if kid in [9000,9010]:
            caps = [r for r in large if r['kid'] == kid]
            for r in caps:
                assert all(r['kernel'][k] == baseline_private[symbol][k] for k in ['instruction_sha256','metadata','descriptor_normalized_sha256'])
            row.update(primary='shape_dependent', secondary=['compute','memory_latency'],confidence='中等；按阶段判断',
                       classification_scope='原版自身短/长K代表；不是所有winner统一标签',representatives=[r['shape'] for r in caps],
                       finding='短K：启动与写回固定工作占大头，SMEM/scale依赖是memory latency贡献；长K：MFMA发射/依赖占主要局部时间，倾向compute/issue，同时有LDS/VMEM供数成本。未证明全卡计算峰值。')
            row['evidence'] = [record_evidence(OUT / 'att_analysis.json', [r['capture'] for r in caps], kernel, baseline_private[symbol])]
            row['phase_metrics'] = [dict(shape=r['shape'], summary=r['summary']) for r in caps]
            row['optimization_state'] = 'original_retained_with_separate_candidate'
            for r in caps:
                name = f'target{r["index"]}_kid{kid}_sq_ea'
                own, epoch = owner_by_name[name]; assert epoch['strict_clean']
                path = OUT / 'baseline_diagnostics' / name / 'parsed.json'
                row['evidence'].append(record_evidence(path, [], kernel))
                row.setdefault('counter_metrics', []).append({'shape':r['shape'], 'metrics':read(path)['groups'][0]['median_metrics'],
                                                              'strict_owner_command':name,'claim_sha256':own['sha256']})
        elif kid == 9021:
            row.update(primary='shape_dependent',secondary=['memory_latency','compute'],confidence='scale因果较强；总瓶颈中等',
                       classification_scope='短K采用版ATT与前后Event；长K为采用前自身baseline',
                       representatives=[[1024,7168,384],[1024,7168,16384]],
                       finding='短K的scale请求串行与LDS发布等待是已验证贡献，重叠请求后全42winner GM+1.5756%；长K仍混合matrix issue、LDS和panel refill。不能据此称采用版全部latency瓶颈已消除。',
                       optimization_state='prior_scale_issue_publish_retained')
            # Read the original shape directly from the retained plan.
            plan = read(RESUME / 'diagnostics/plan.json')
            row['representatives'] = [t['shape'] for t in plan['targets'] if t['kid']==9021][:2]
            row['evidence'] = [record_evidence(RESUME / 'diagnostics/scale_issue_publish_att_analysis.json', [], kernel),
                               record_evidence(RESUME / 'diagnostics/att_analysis.json', [], kernel),
                               record_evidence(RESUME / 'scale_issue_publish/retention_decision.json', [], kernel)]
        elif symbol in merged:
            caps = merged[symbol]
            image = caps[0]['identity']['code_object']; source = next(r for r in elf.COMMON.summarize_image(Path(image).read_bytes()) if r['name']==symbol)
            row['evidence'] = [record_evidence(OLD / 'diagnostics/merged_att_analysis.json', [r['capture_name'] for r in caps], kernel, source)]
            row['representatives'] = [r['shape'] for r in caps]
            row['phase_metrics'] = [dict(shape=r['shape'],summary=r['wave_statistics'],phases=r['phase_clock_statistics']) for r in caps]
            row['confidence'] = '中等；自身局部ATT'
            row['classification_scope'] = '自身代表形状；' + ('已采用改动之前的baseline机制' if kernel['instruction_sha256']!=source['instruction_sha256'] else '与最终device相同')
            if kid == 9020:
                if inv['K_values']==[384]:
                    row.update(primary='memory_latency',secondary=['setup_output','compute'],finding='fixed384启动publication/scale producer延迟与固定输出成本显著；采用请求重排，全7winner GM+2.5715%。采用版总瓶颈未重新采集。')
                elif '128, 128' in traits:
                    row.update(primary='memory_latency',secondary=['compute','memory_issue'],finding='128x128 runtime代表VMEM依赖约12026、MFMA事件12406 clocks，另有LDS/barrier；供数与计算混合，不能套192x256长runtime结论。')
                elif inv['K_values']==[768]:
                    row.update(primary='mixed',secondary=['memory_latency','compute','setup_output'],finding='fixed768启动5188/总22338、after-last4924 clocks，MFMA4456与barrier2822；设置/输出与计算供数混合。')
                else:
                    row.update(primary='mixed',secondary=['compute','memory_latency','memory_issue'],finding='稳态MFMA发射与matrix VMEM issue、LDS/barrier共同占时；倾向compute/issue加供数，尚无绝对执行管线饱和证据。固定K和runtime各用自己的ATT。')
            elif kid == 9022:
                row.update(primary='shape_dependent',secondary=['memory_latency','compute','memory_issue'],finding='短K启动5016/总10578 clocks，SFA→wait/store→SFB串行；长K混合MFMA、matrix issue、LDS及panel32 refill。全局候选虽GM微正，有6个5/5 loser而拒绝。')
            else:
                row.update(primary='mixed',secondary=['compute','memory_latency','memory_bandwidth_candidate'],finding='大输出的计算/供数/写回混合；barrier cohort到达差908 clocks，而last-arrival到release仅4。clean DRAM4.267GB/2.923TB/s确认输出流量重要，不能证明HBM饱和；midpoint全10winner失败。')
                path = OLD / 'diagnostics/target13_kid9030_sq_ea_recollect/parsed.json'
                row['evidence'].append(record_evidence(path,[],kernel)); row['counter_metrics'] = read(path)['groups'][0]['median_metrics']
        elif symbol in narrow:
            caps = narrow[symbol];image = caps[0]['identity']['captured_code_object']
            source = next(r for r in elf.COMMON.summarize_image(Path(image).read_bytes()) if r['name']==symbol)
            row.update(primary='memory_latency',secondary=['compute','memory_issue'],confidence='机制中等；采用前baseline',
                       classification_scope='已采用改动之前自身baseline；不宣称采用后独占主瓶颈',representatives=[r['shape'] for r in caps])
            row['evidence'] = [record_evidence(OLD / 'narrow_att_analysis.json', [r['target_index'] for r in caps],kernel,source)]
            row['phase_metrics'] = [dict(shape=r['shape'],summary=r['wave_summary']) for r in caps]
            row['finding'] = ('runtime对齐refill VMEM串行、发布依赖；动态readlane每wave仅6次用于remainder/output，静态spill不是稳态120次读。采用版全4winner GM+1.1173%。' if kid==9023 else 'fixed7168启动约15.5%，scale producer/publication延迟是贡献，稳态计算与LDS issue仍重要。采用版全9winner GM+1.2745%。')
        elif symbol in old_small:
            r = old_small[symbol]
            image = r['identity']['image']['path'];source = next(v for v in elf.COMMON.summarize_image(Path(image).read_bytes()) if v['name']==symbol)
            assert all(source[k]==kernel[k] for k in ['instruction_sha256','metadata','descriptor_normalized_sha256'])
            row.update(primary='memory_latency',secondary=['setup_output'],confidence='中等；自身局部ATT',
                       classification_scope='自身代表；最终device身份相同',representatives=[r['shape']])
            row['evidence'] = [record_evidence(OLD / 'small_att_analysis.json',[r['capture']],kernel,source)]
            row['phase_metrics'] = r['summary']
            findings = {
                9040:'短K startup2452/3224 clocks约76%，主要SMEM参数/scale供给与固定设置；小grid不等于dispatch已证实。',
                9042:'实际9071 fixed7168 register队列有VMEM请求/依赖等待；这是9071自身证据，不能代替9073或runtime。',
                9046:'cluster2 MFMA间隔648 vs普通316 clocks，VMEM wait约20；偏local LDS依赖/同步和issue，未证明LDS带宽饱和。',
                9047:'短K startup约66%，SMEM wait约1019 clocks；设置/内存依赖占主，未证明dispatch。',
                9051:'M1代表VMEM请求issue/返回等待显著；旧4组clean PMC约5.962TB/s、LFIFO24.03%、UTCL1在途6.18%，支持memory request/latency压力，可能有bandwidth成分；M方向16x padding浪费。',
                9055:'scale producer让startup barrier到达延后约2300 clocks，SFA/SFB wait892/944；稳态仍有deep-ring LDS issue。',
                9062:'实际80/fixed16384/split2 producer scale尾使startup到达延后2308 clocks；steady local LDS/同步。完整API含matching reducer，producer分类不能代替完整API。'}
            row['finding'] = findings[kid]
            if kid==9051:
                for folder in sorted((ROOT/'reports/opus_bound_analysis_20261007/char_profiles').glob('char_target31_kid9051_*')):
                    path=folder/'parsed.json'
                    if path.exists():row['evidence'].append(record_evidence(path,[],kernel))
            if kid in [9042,9051]:row['secondary']=['memory_issue','memory_bandwidth_candidate']
            elif kid in [9046,9055,9062]:row['secondary']=['local_LDS_issue','compute','synchronization']
        elif symbol in additional:
            r = additional[symbol];s = r['summary'];main = s['main_classes'];startup = 100*s['startup']['median']/s['life']['median']
            assert all(r['kernel'][k]==kernel[k] for k in ['instruction_sha256','metadata','descriptor_normalized_sha256']) and r['strict_owner']['strict_clean']
            register = 'small_register' in kernel['demangled']
            secondary = ['memory_issue','setup_output'] if register else ['local_LDS_issue','compute','synchronization']
            row.update(primary='memory_latency',secondary=secondary,confidence='中等；自身局部ATT',
                       classification_scope='自身代表；最终device身份相同',representatives=[r['target']['shape']])
            row['evidence'] = [record_evidence(OUT / 'remaining_att_analysis.json',[r['capture']],kernel,r['kernel'])]
            row['strict_owner'] = r['strict_owner'];row['phase_metrics']=s;row['MFMA_count_validation']=r['MFMA_count_validation']
            dep = main.get('VMEM_dependency' if register else 'LDS_dependency',0)
            issue = main.get('VMEM_issue' if register else 'LDS_issue',0)
            mfma = main.get('MFMA_issue_or_dependency',0)
            row['finding'] = f'启动到first MFMA占局部寿命{startup:.1f}%；main '+('VMEM' if register else 'LDS')+f'依赖{dep:g}、issue{issue:g}，MFMA事件{mfma:g} clocks。'+('请求供给/返回等待主导，未证明外部带宽饱和。' if register else 'local LDS供数/同步与issue为主要线索，未证明LDS带宽饱和。')
            if not register and dep < mfma and issue+dep < mfma*1.5:
                row['primary']='mixed';row['finding']+=' 计算与local供数共同重要。'
            if kid==9054:row['finding']+=' K768四wave计数为8/8/16/16，总48；12仅均值，不作为每wave期望。'
        else:
            if symbol.startswith('_Z43opus_gemm_mxscale_bpreshuffle_reduce_kernel'):
                row['role']='reducer';row['finding']='保留身份与完整调用支持；没有每个reducer独立的带宽/latency/dispatch诊断。'
                row['secondary']=['memory_latency_hypothesis','memory_bandwidth_hypothesis']
            elif kid in [9023,9024]:row['finding']='合法support/control body，无历史winner；数值或control Event不构成本body独立瓶颈分类。'
            else:row['finding']='无历史winner的兼容producer；保留身份，不能借同parent其他配置的ATT给它定性。'
        rows.append(row)

assert len(rows)==56 and len({r['parent_id']for r in rows})==26
assert sum(r['historical_winner_count']for r in rows)==745
assert sum(r['role']=='winner_producer'for r in rows)==47
assert all(r['evidence'] for r in rows if r['role']=='winner_producer')

# Split alternatives are separate final entries, with no fabricated final ATT.
final_manifest = OUT / 'split_formal_scale_reset/integration_manifest.json'
if final_manifest.exists():
    selected = read(final_manifest)
    for alt in selected['new_candidates']:
        base = next(r for r in rows if r['parent_id']==alt['base_parent_id'])
        row = dict(base)
        row.update(parent_id=alt['kid'],symbol=alt['kernel']['name'],traits=alt['kernel']['demangled'],actual_ids=[alt['kid']],
                   historical_winner_count=0,role='new_tuning_candidate',primary='unconfirmed',confidence='最终候选瓶颈未重采',
                   classification_scope='原版诊断用于选方向；优化版只有自身性能/数值和machine身份',
                   finding=('SFA raw临时变量清零；VGPR477→469。' if alt['kid']==9001 else '循环展开4；VGPR497→492。')+'原版的短/长K类型不能直接当作本候选最终主瓶颈。'+alt['decision_reason'],
                   evidence=[record_evidence(final_manifest,[],alt['kernel'])],kernel_identity=alt['kernel'],
                   optimization_state='applied_separate_tuning_candidate',secondary=['compute_hypothesis','memory_latency_hypothesis'])
        row.pop('phase_metrics',None);row.pop('counter_metrics',None);rows.append(row)
        row['representatives']=[r['shape'] for r in read(OUT/'split_formal_scale_reset/api_results.json')['rows']if r['kid']==alt['kid']]

categories = {'compute':'计算执行或计算指令issue限制，需要定位具体管线',
              'memory_bandwidth':'指定内存层级的实际带宽接近该负载可达值，并有相应回压',
              'memory_latency':'VMEM/SMEM/LDS数据依赖或供应等待；本报告明确标明层级',
              'dispatch':'前端/资源分配使CU缺少可运行wave，需要动态并发与SPI等证据'}
data = {'status':'finite_all_public_entries_classified_with_explicit_unknowns',
        'generated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'reference_document':{'path':str(DOC),'sha256':sha(DOC),'section':'4.2 and 4.2.1–4.2.5'},
        'categories':categories,'public_parents':len({r['parent_id']for r in rows}),'device_entries':len(rows),
        'original_public_parents':26,'original_device_entries':56,'original_winner_configurations':47,
        'historical_winner_assignments':745,'small_own_ATT_configs':33,'new_small_ATT_configs':26,
        'unconfirmed_original_entries':9,'rows':rows,
        'limits':['Specific shape/path/phase classification; parent is not universally one bound type.',
                  'Local memory issue pressure is not proof of external HBM or LDS bandwidth saturation.',
                  'No exclusive HBM-bandwidth bound or dispatch-bound result is proven by the retained evidence.',
                  'ATT queue membership does not measure each request return latency.',
                  'Shader-clock category medians are correlated, not wall-time shares or independent samples.',
                  'GRBM denominators and occupancy are uncalibrated; no absolute utilization or cycle-to-ns claims.',
                  'Adopted 9020/9023/9024 changes use pre-change mechanism evidence; current dominant bound unconfirmed.']}
(OUT/'bound_classification.json').write_text(json.dumps(data,indent=2)+'\n')
columns=['parent_id','actual_ids','role','traits','historical_winner_count','primary','secondary','confidence','classification_scope','representatives','finding','optimization_state','symbol']
with (OUT/'bound_classification.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader()
    for r in rows:writer.writerow({k:json.dumps(r[k],ensure_ascii=False) if isinstance(r[k],(list,dict)) else r[k] for k in columns})

text = ['# 全系列 bound 分类：按实际配置、形状和阶段', '',
        f'原有 **26 个公开 parent / 56 个 device entry** 均已列出，含47个历史winner producer配置、四个support producer和五个reducer。本次新增候选计入后为 **{data["public_parents"]} parent / {len(rows)} entry**。745为历史winner归属数，不是本次745次性能重测。', '',
        '最新依据为[带宽文档第4.2节](/root/workspace/trans_github/mi300_gpu_bandwidth_chapters3_4_5_notes.md:789)。结论是：长K大tile更偏compute/issue与供数混合，register队列多偏VMEM请求/依赖，LDS/fine多偏local LDS依赖与同步；短K常有明显启动与输出固定成本。**现有证据没有确认任何整个系列属于纯HBM bandwidth bound或dispatch bound。**', '',
        'memory latency在表中标注VMEM、SMEM或LDS；它表示供数依赖贡献，不等于已测出单次HBM返回延迟。LDS issue多也不能自动定为LDS bandwidth饱和。compute/issue是方向判断，尚未由校准全卡利用率证明达到计算峰值。', '',
        '分类证据与优化收益分别记录：同一个parent可以随K、grid和实际body变化；采用前baseline的ATT只说明优化贡献，不能冒称采用后的主瓶颈仍完全相同。新增9001/9011采用版没有重采ATT，新增行的代表是正式API验证尺寸。', '',
        '9040+的33个实际winner配置均已有自身ATT：旧7个加本次补26个，补采逐项核对CO/FUNC/full metadata/normalized descriptor和严格GPU owner；每个配置只采一个自身代表。[新26项分析](remaining_att_analysis.json)、[旧7项分析](../opus_remaining_20261008/small_att_analysis.json)。', '',
        '## 按公开 parent 速查', '',
        '| parent | 原始/候选配置数 | 类型判断 | 关键机制与适用范围 |', '| --- | ---: | --- | --- |']
parent_notes = {
    9000:('长K compute/issue；短K memory latency贡献＋setup/output','原版3个自身代表；新9001分别选型'),
    9010:('长K compute/issue＋LDS；短K memory latency贡献＋setup/output','原版2个自身代表；新9011分别选型'),
    9020:('按body：fixed384 latency；其他compute/供数混合','7个body各有ATT；128x128 runtime更偏VMEM等待'),
    9021:('短K scale latency；长K compute/供数混合','scale请求与publish重叠已验证贡献，采用后长K未重采'),
    9022:('短K memory latency；长K混合','SFA/SFB串行、panel32 refill、matrix/LDS issue'),
    9023:('runtime memory latency贡献＋compute/issue','采用前refill VMEM机制；fixed support未确认'),
    9024:('fixed7168 memory latency贡献＋compute/issue','启动约15.5%；runtime control未确认'),
    9030:('compute＋memory latency＋输出服务混合','4.267GB/2.923TB/s不是HBM饱和证据'),
    9040:('memory latency＋启动设置','短K约76% startup；不是dispatch证据'),
    9041:('memory latency / VMEM request issue','runtime与实际9050各自ATT'),
    9042:('memory latency / VMEM request issue','runtime、实际9071/9073分开判断'),
    9043:('memory latency贡献 / local LDS issue','短K约42% startup'),
    9044:('memory latency / local LDS；部分compute混合','runtime与实际9056各自ATT'),
    9045:('memory latency / local LDS＋issue混合','LDS依赖7562、issue8736 clocks'),
    9046:('memory latency / local LDS同步＋issue','cluster2 gap648 vs316；非LDS带宽饱和结论'),
    9047:('memory latency＋启动设置','短K约66% startup'),
    9049:('memory latency / local LDS＋启动设置','短K约47% startup'),
    9051:('memory request/latency压力；可能带宽成分','5.962TB/s与LFIFO/UTCL1线索；未证明独占HBM带宽限制'),
    9052:('memory latency / VMEM request issue','runtime与实际9070各自ATT'),
    9053:('短K memory latency＋setup；fixed VMEM issue','runtime与实际9072分开'),
    9054:('短K memory latency＋setup','startup约65%；wave计数8/8/16/16'),
    9055:('memory latency / scale发布与local LDS','scale producer启动到达差约2300 clocks'),
    9060:('memory latency / local LDS＋compute混合','runtime/K3072/K7168各自ATT'),
    9061:('memory latency / local LDS与同步','main LDS依赖15666、barrier7030 clocks'),
    9062:('producer memory latency / local LDS与同步','两winner body各自ATT；另兼容producer与2reducer未独立分类'),
    9063:('producer memory latency / local LDS＋compute混合','7winner body各自ATT；兼容producer与3reducer未独立分类'),
    9001:('未确认优化后主瓶颈','SFA raw清零减少VGPR；自身性能独立；原版ATT只作方向'),
    9011:('未确认优化后主瓶颈','循环展开4；自身性能独立；原版ATT只作方向')}
for kid in sorted({r['parent_id']for r in rows}):
    kind,note=parent_notes[kid];text.append(f'| {kid} | {sum(r["parent_id"]==kid for r in rows)} | {kind} | {note} |')
text += ['', '## 全部实际 entry', '',
         '完整symbol、machine身份、owner与原始报告哈希见[JSON](bound_classification.json)；可筛选表见[CSV](bound_classification.csv)。`未确认`是完整清单中的明确状态，不借别的body测量填标签。', '',
         '| # | parent / actual | traits | 自身代表 M×N×K | 类型与置信范围 | 关键依据 |',
         '| --- | --- | --- | --- | --- | --- |']
for i,r in enumerate(rows,1):
    traits=r['traits'].replace('opus_gemm_mxscale_bpreshuffle_','').replace('opus_gemm_small_','').replace('_traits_gfx950','')
    shape='；'.join('×'.join(map(str,s)) for s in r['representatives']) or '—'
    evidence='；'.join(f'[{Path(e["path"]).name}]({"../../"+e["path"] if e["path"].startswith("reports/") else e["path"]})' for e in r['evidence'][:1])
    text.append(f'| {i} | {r["parent_id"]} / {",".join(map(str,r["actual_ids"]))} | `{traits}` | {shape} | {r["primary"]}；{r["confidence"]}；{r["classification_scope"]} | {r["finding"]} {evidence} |')
text += ['', '## 证据边界与后续优先级', '',
         'Memory bandwidth需要同层级的可达带宽、请求/通道分布与回压；多数代表缺少完整的匹配窗口。9051外部请求压力较强，仍无法把LFIFO/翻译队列与HBM带宽限制唯一分开。9030写流量大，但仅凭bytes/time不能定为bandwidth bound。', '',
         'Dispatch需要动态活跃wave、SPI/资源分配stall和完成时间影响。静态VGPR/LDS大、224个WG少于256CU、短K启动长，均只提供线索；当前没有可靠dispatch判定。CPU注册/分支身份核对与dispatch瓶颈测量是不同事情。', '',
         '下一步若继续针对机制优化，register优先查请求并发/等待位置，LDS/fine优先查供数依赖与barrier到达差，大tile长K优先查MFMA issue与matrix ring重叠；每项都需要自身完整调用性能决定。当前分类所需的有限采集已收口，不把全系列分类写成全系列已提速。', '']
(OUT/'BOUND_CLASSIFICATION.md').write_text('\n'.join(text))
print(json.dumps({k:data[k]for k in ['public_parents','device_entries','original_winner_configurations','small_own_ATT_configs','unconfirmed_original_entries']}))
