#!/usr/bin/env python3
"""Build final23parent/44config report from applied root progress without mutating it."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ref(path):
    return {'path': str(Path(path).resolve()), 'sha256': sha(path)}


def link(label, target):
    path = Path(target)
    if path.is_absolute():
        try:
            path = path.relative_to(HERE)
        except ValueError:
            return f'[{label}]({path})'
    assert (HERE / path).exists(), str(path)
    return f'[{label}]({path.as_posix()})'


def trait_short(traits):
    aliases = [('opus_gemm_mxscale_bpreshuffle_8wave_192x256_large_output_traits_gfx950', 'large_output'),
               ('opus_gemm_mxscale_bpreshuffle_8wave_traits_gfx950', 'merged8'),
               ('opus_gemm_mxscale_bpreshuffle_4wave_160x128_traits_gfx950', 'merged160'),
               ('opus_gemm_mxscale_bpreshuffle_4wave_64x128_traits_base_gfx950', 'narrow64x128'),
               ('opus_gemm_mxscale_bpreshuffle_4wave_64x64_traits_base_gfx950', 'narrow64x64'),
               ('opus_gemm_small_register_traits_gfx950', 'register'),
               ('opus_gemm_small_lds_traits_gfx950', 'lds'),
               ('opus_gemm_mxscale_bpreshuffle_fine_traits_gfx950', 'fine')]
    for original, short in aliases:
        if traits.startswith(original):
            return traits.replace(original, short, 1)
    raise AssertionError(traits)


def state_text(entry):
    state = entry['current_optimization_state']
    if state.startswith('adopt_'):
        return '采用候选；已应用并验证'
    if entry.get('decision') == 'retain_prior_accepted_runtime_reuse_alias':
        return '保留此前已采用runtime alias'
    if entry.get('decision') == 'reject_finite_candidate_keep_current' or 'reject' in state or 'rejected' in state:
        return '拒绝本轮候选；保留原配置'
    return 'keep自身当前配置'


def evidence_links(entry):
    parent = entry['parent_id']
    if parent == 9020:
        return link('screen', 'scale_issue/screen_analysis.json') + '、' + link('取舍', 'scale_issue/retention_decision.json')
    if parent == 9022:
        return link('166覆盖', 'scale_issue/coverage_independent_review.json') + '、' + link('取舍', 'scale_issue/retention_decision.json')
    if parent in [9023, 9024]:
        return link('13winner取舍', 'narrow_candidate/retention_recommendation.json')
    if parent == 9030:
        return link('10覆盖', 'large_midpoint/coverage_independent_review.json') + '、' + link('拒绝', 'large_midpoint/global_decision.json')
    current = entry['own_this_round_candidate_Event_actualwinner_shapes']
    if current:
        actual = entry['actual_configuration_ids'][0]
        path = 'register_issue_order/coverage_analysis.json' if actual == 9051 else ('fine_startup/results_analysis.json' if actual == 9062 else 'ring_overlap/results_analysis.json')
        return link('自身Event', path) + '、' + link('配置/历史证据', 'small_family_review.json')
    return link('自身配置/历史范围', 'small_family_review.json') + '、' + link('自身数值门', 'small_config_gate_audit.json')


def performance_scope(entry, inventory):
    parent = entry['parent_id']
    if parent == 9020:
        if entry['current_optimization_state'].startswith('adopt_'):
            return '全部7/7自身winner Event；另1screen（去重）'
        row = entry['this_round_candidate_performance'][0]
        assert len(entry['this_round_candidate_performance']) == 1
        own = row['shape'] in inventory['winner_shapes']
        return '1个自身配置screen Event ' + ','.join(map(str, row['shape'])) + ('（自身winner）' if own else '（支持域代表）') + '；未覆盖全部winner'
    if parent == 9022:
        return '全部159/159自身winner Event'
    if parent == 9023:
        return '全部4/4自身winner Event；另M16非winner'
    if parent == 9024:
        return '全部9/9自身winner Event；另unchanged runtime control'
    if parent == 9030:
        return '全部10/10自身winner Event（8地址池）；另screen/2非winner'
    current = entry['own_this_round_candidate_Event_actualwinner_shapes']
    if current:
        count = len(current)
        scope = f'{count}/{entry["historical_winner_count"]}自身winner候选Event'
        if entry['actual_configuration_ids'] == [9062]:
            scope += '，含matching split2 reducer'
        elif count == 1:
            scope += ' ' + ','.join(map(str, current[0]))
        return scope
    historical = entry['own_historical_exactconfig_Event_row_count']
    local = entry['own_exact_local_ATT_measured']
    return '本轮新候选性能未测' + ('；自身ATT1形状' if local else '') + (f'；自身历史exact-config Event {historical}行' if historical else '')


def scope_limit(entry):
    parent = entry['parent_id']
    if parent == 9020:
        return ('仅existing fixed384；保留round2时间+19.815%异常' if entry['current_optimization_state'].startswith('adopt_')
                else '单代表mixed；不从其他body失败冒称自身全winner性能')
    if parent == 9022:
        return 'GM+0.821%仍有6winner稳定5/5更慢；全局拒绝，无新K阈值'
    if parent == 9023:
        return '保留longK winner median成本+0.542%、M16+0.417%；VGPR232→251/lane spill46→48'
    if parent == 9024:
        return '9winner45/45pair更快；仅原fixed7168，runtime不变'
    if parent == 9030:
        return '10median全负、4/50pair快；22%barrier波份额不是可删墙钟'
    if entry['own_this_round_candidate_performance_measured']:
        actual = entry['actual_configuration_ids'][0]
        if actual == 9051:
            return '17项8正8退1平，仅3项5/5快；统一reject，不按shape切scope'
        return '自身有限候选失败；不宣称全部winner或别的配置有性能结果'
    if entry.get('decision') == 'retain_prior_accepted_runtime_reuse_alias':
        return '采用依据为此前自身winner Event；本轮仅保留，未新增采用'
    if entry['actual_configuration_ids'][0] in [9071, 9073]:
        return '自身旧N32失败保留N48；9071草稿未build，9071ATT不是9073性能'
    return '自身type/source/identity/数值及存在时历史证据支持keep；共享机制不算自身性能'


def main():
    progress_path = HERE / 'family_progress.json'
    readonly = [progress_path, HERE / 'README.md',
                ROOT / 'reports/opus_bound_analysis_20261007/COMPUTE_BOUND_ANALYSIS_AND_OPTIMIZATION.md',
                ROOT / 'reports/opus_bound_analysis_20261007/MEMORY_BOUND_ANALYSIS_AND_OPTIMIZATION.md']
    before = {str(path): sha(path) for path in readonly}
    progress = json.loads(progress_path.read_text())
    inventory = json.loads((HERE / 'inventory.json').read_text())
    integration_path = HERE / 'formal_selected/integration_manifest.json'
    integration = json.loads(integration_path.read_text())
    api_path = HERE / 'formal_selected/api_gate/gpu_api_analysis.json'
    api = json.loads(api_path.read_text())
    identity = json.loads((HERE / 'formal_selected/identity_audit.json').read_text())
    gate = json.loads((HERE / 'diagnostics/all_config_gate_results.json').read_text())
    assert progress['status'] == integration['status'] == 'applied_verified_not_committed'
    assert progress['pending_experiments'] == 0
    assert progress['selected_entries_count'] == integration['changed_device_entries'] == 3
    assert api['status'] == 'passed_official_API_signed8_loaded_identity_all26parents'
    assert api['target_count'] == len(api['rows']) == 44 and api['public_parent_count'] == 26
    assert sum(api['numerical_calls_by_label'].values()) == 472
    assert gate['status'] == 'passed' and len(gate['rows']) == 44
    assert gate['official_binary_sha256'] == inventory['official_module']['sha256']
    assert identity['candidate_binary_sha256'] == api['official_module']['sha256'] == integration['official_module']['sha256']
    assert sha(api['official_module']['path']) == api['official_module']['sha256']
    rows = progress['entries']; baseline = {row['symbol']: row for row in inventory['entries']}
    assert len(rows) == len(baseline) == 44 and {row['symbol'] for row in rows} == set(baseline)
    assert sum(row['historical_winner_count'] for row in rows) == 536
    assert len({row['parent_id'] for row in rows}) == 23
    adopted = [row for row in rows if row['current_optimization_state'].startswith('adopt_')]
    assert len(adopted) == 3 and {row['parent_id'] for row in adopted} == {9020, 9023, 9024}
    scale = json.loads((HERE / 'scale_issue/retention_decision.json').read_text())
    narrow = json.loads((HERE / 'narrow_candidate/retention_recommendation.json').read_text())
    large = json.loads((HERE / 'large_midpoint/global_decision.json').read_text())
    assert len(scale['9022_five_of_five_slower_winners']) == 6
    assert sum(row['kid'] == 9023 and row['actual_winner'] for row in narrow['rows']) == 4
    assert sum(row['kid'] == 9024 and row['actual_winner'] for row in narrow['rows']) == 9
    assert large['decision'] == 'reject_global9030_midpoint'
    text = '# 剩余23个parent与44个实际配置的最终结果\n\n'
    text += ('当前状态为 **applied_verified_not_committed**：本轮新增采用并应用的device entry只有9020原fixed384、9023原runtime和9024原fixed7168，共三个源码文件。'
        '其余配置保留当前selected，已测候选按自身证据拒绝；此前已采用9021与9042/9053/9054 runtime B-scale alias保持。'
        '没有commit或push，待执行实验为0。逐symbol状态依据 [family_progress.json](family_progress.json)，源码应用和验证依据 [integration_manifest.json](formal_selected/integration_manifest.json)。\n\n')
    text += ('本表的44项是与536个历史winner相交的**实际producer symbol配置**，分属23个public parent；同ID的runtime/fixedK或不同traits单独成行。'
        '它们与正式构建的26parent/56entry、正式API的26parent/44target分别计数。全部44配置各取一个自身winner完成原Oct8 baseline signed2repeat/reference/guard；'
        '最终正式模块再完成44 API target、352次official加120次private，共472次数值调用。两组数值门都没有性能结论，未测配置的keep不表示全winner提速。\n\n')
    text += f"[最终正式模块](jit_formal_selected/module_deepgemm_opus.so)的SHA256为 `{api['official_module']['sha256']}`。"
    text += ('[正式身份审计](formal_selected/identity_audit.json)确认56entry中3改变精确匹配已测候选、53不变；313生成文件、206build object、202linked gfx950 bundle/263device entry核对通过。'
        '[正式API审查](formal_selected/api_gate/gpu_api_analysis.json)确认44target均通过signed8/reference/repeat/guard、实际加载SHA和物理GPU owner；'
        '[原44配置数值门](diagnostics/all_config_gate_results.json)用于各配置baseline正确性。small为33个winner-producer配置、另外2个无历史winnerproducer及5个reducer；'
        '本轮small候选性能范围严格为4config/21winner、ATT为7config、各自数值门为33config，5reducer保留身份且仅matching调用计时按实际包含。\n\n')
    text += '## 23个public parent汇总\n\n'
    text += '| parent | 实际配置数 | 历史winner数 | 本轮候选性能实测范围 | 最终状态 | 依据 |\n| --- | ---: | ---: | --- | --- | --- |\n'
    parent_specs = {
        9020: ('fixed384全部7winner；其余6body各1screen Event', '采用原fixed384；其余6配置保留', 'scale_issue/retention_decision.json'),
        9022: ('全部159winner Event', '拒绝global候选；保留原配置', 'scale_issue/retention_decision.json'),
        9023: ('全部4winner Event；M16非winner', '采用原runtime；fixed支持域不变', 'narrow_candidate/retention_recommendation.json'),
        9024: ('全部9winner Event；unchanged runtime control', '采用原fixed7168；runtime不变', 'narrow_candidate/retention_recommendation.json'),
        9030: ('全部10winner Event；8地址池，另screen/2非winner', '拒绝midpoint；保留原配置', 'large_midpoint/global_decision.json')}
    for parent in progress['scope_parent_ids']:
        children = [row for row in rows if row['parent_id'] == parent]
        winners = sum(row['historical_winner_count'] for row in children)
        if parent in parent_specs:
            scope, state, path = parent_specs[parent]
        else:
            count = sum(len(row['own_this_round_candidate_Event_actualwinner_shapes']) for row in children)
            scope = f'{count}个自身winner候选Event' if count else '本轮新候选性能未测；自身数值/identity/存在时历史证据'
            state = ('拒绝自身有限候选；保留原配置' if count else
                     '保留此前runtime alias；其余keep' if any(row.get('decision') == 'retain_prior_accepted_runtime_reuse_alias' for row in children)
                     else 'keep自身当前配置')
            path = 'register_issue_order/coverage_analysis.json' if parent == 9051 else ('fine_startup/results_analysis.json' if parent == 9062 else 'ring_overlap/results_analysis.json' if parent in [9046,9055] else 'small_family_review.json')
        text += f'| {parent} | {len(children)} | {winners} | {scope} | {state} | {link("记录", path)} |\n'
    text += ('\n采用三entry的等shape geomean收益为9020 fixed384 **+2.571542%**、9023 runtime **+1.117277%**、9024 fixed7168 **+1.274511%**。'
        '9020的`12288,7168,384` round2时间增加19.815%仍保留，原因未确认；9023的`544,7168,16384` winner median时间增加0.542%、M16非winner增加0.417%，'
        'VGPR232→251和scalar lane spill46→48风险也保留。9022虽整体GM+0.821%，六winner稳定5/5更慢；9030全十median负；9051全17为8正8退1平，统一拒绝。'
        '这些有限集合统计不是生产batch或全部支持域收益。\n\n')
    text += '## 全部44个actual producer配置\n\n'
    text += ('短traits标签仅省略类型固定前缀：`merged8`、`merged160`、`narrow64x128`、`narrow64x64`、`large_output`、`register`、`lds`、`fine`对应原traits家族，尖括号内参数原样保留。'
        '完整mangled symbol、selected traits、状态原文和所有winner shape见 [family_progress.json](family_progress.json) 与 [inventory.json](inventory.json)。'
        '9042/9053/9054旧inventory descriptive false标签已按实际selected symbol/SHA纠正为true，历史文本单列保存。\n\n')
    text += '| # | parent | actual ID | selected traits短标签 | 历史winner数 | 最终state | 性能实测范围 | 证据与适用限制 |\n| --- | --- | --- | --- | ---: | --- | --- | --- |\n'
    for index, row in enumerate(rows, 1):
        item = baseline[row['symbol']]
        assert row['historical_winner_count'] == item['winner_count']
        traits = trait_short(row['traits'])
        text += f'| {index} | {row["parent_id"]} | {",".join(map(str,row["actual_configuration_ids"]))} | `{traits}` | {row["historical_winner_count"]} | {state_text(row)} | {performance_scope(row,item)} | {evidence_links(row)}；{scope_limit(row)} |\n'
    text += ('\n本轮新候选有完整自身winner集合性能的配置是9020 fixed384、9022、9023 runtime、9024 fixed7168、9030、runtime9051和fixed9062；'
        '9046/9055各只测一个自身winner，fixed9062测其两个winner，其余9020 body各只一个screen。其余small配置未测本轮新候选性能，'
        '自身历史exact-config Event或局部ATT分别列出，共享机制说明和别的配置reject不算其性能实测。\n\n')
    text += ('44配置表不包含没有历史winner的合法support body：9023 fixed7168、9024 runtime，以及small的两个兼容/runtime producer；'
        '正式56entry由本表44配置、这四body、5个reducer和范围外已冻结9000/9010/9021三个entry组成，均逐项检查身份。5reducer无逐个独立性能声明，fixed9062 initial handoff的完整Event包含实际matching split2 reducer；'
        'split1的9046/9055/9051包含producer内部处理，不因runner通用includes文字额外算一次reducer。'
        '所有local ATT与counter仍限自身shape/CO/PC/物理owner；GRBM未校准，不声明绝对MFMA利用率、动态occupancy或shader clocks→ns。\n\n')
    text += '生成依据SHA256：\n\n'
    for label, path in [('family_progress',progress_path),('inventory',HERE/'inventory.json'),
                         ('integration',integration_path),('正式API',api_path),
                         ('baseline44gate',HERE/'diagnostics/all_config_gate_results.json')]:
        text += f'- {link(label, path)}：`{sha(path)}`\n'
    output = HERE / 'FAMILY_RESULTS.md'
    output.write_text(text)
    after = {str(path): sha(path) for path in readonly}
    assert before == after
    all_links = re.findall(r'\]\(([^)]+)\)', text)
    assert all((Path(target) if Path(target).is_absolute() else HERE / target).exists() for target in all_links)
    audit = {'status':'passed_CPU_final23parent44config_FAMILY_RESULTS_scope_links_audit',
        'generated_utc':datetime.now(timezone.utc).isoformat(), 'root_progress_README_originaldocs_modified':False,
        'parent_rows':23,'actualconfig_rows':44,'historical_winners':536,'new_adopted_entries':3,
        'new_adopted_parent_ids':[9020,9023,9024], 'root_status':progress['status'],
        'formal_API_targets':44,'formal_API_public_parents':26,'formal_API_numerical_calls':472,
        'baseline_own_config_numerical_gate_targets':44,'small_candidate_Event_configs':4,
        'small_candidate_Event_actualwinner_shapes':21,'all44config_performance_claim':False,
        'all_links_existing':len(all_links),'report':ref(output),'script':ref(Path(__file__)),
        'inputs_unchanged_after_generation':before}
    (HERE/'FAMILY_RESULTS_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit))


if __name__ == '__main__':
    main()
