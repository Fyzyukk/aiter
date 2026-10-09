#!/usr/bin/env python3
"""Update the33-config review using verified finite ATT and isolated CPU audit."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ref(path):
    return {'path': str(path), 'sha256': sha(path)}


def finalize_evidence_scope(review, captures, gate_rows):
    """Keep config-local performance separate from shared mechanisms and numerical identity."""
    old = HERE.parent / 'opus_bound_analysis_20261007'
    fine_path = old / 'fine_timing_summary.json'
    fine = json.loads(fine_path.read_text())
    reg_path = old / 'register_timing_summary.json'
    register = json.loads(reg_path.read_text())
    reuse_path = old / 'register_reuse_scoped/current_winner_coverage.json'
    reuse = json.loads(reuse_path.read_text())
    n32_path = old / 'results/fixed_n32_first.json'
    n32 = json.loads(n32_path.read_text())
    n32_audit_path = old / 'fixed_n32/device_audit.json'
    n32_audit = json.loads(n32_audit_path.read_text())
    n32_base = {k['name']: k for k in n32_audit['libraries'][0]['kernels']}
    coverage_path = HERE / 'register_issue_order/coverage_analysis.json'
    coverage = json.loads(coverage_path.read_text()) if coverage_path.exists() else None
    if coverage:
        assert coverage['status'] == 'rejected_runtime9051_full17_mixed_or_weak_Event'
    finite = {
        'fine': json.loads((HERE / 'fine_startup/results_analysis.json').read_text()),
        'ring': json.loads((HERE / 'ring_overlap/results_analysis.json').read_text())}
    n64_path = old / 'n64_clean_summary.json'
    n64 = json.loads(n64_path.read_text())
    for config in review['configurations']:
        processing = config['processing']
        symbol = config['symbol']
        own_shapes = {tuple(row['shape']) for row in config['winner_shapes']}
        historical = []
        shared = []
        if config['family'] == 'fine_lds_split':
            exact = [row for row in fine['rows'] if row['canonical_traits_type'] == config['actual_selected_traits']]
            for row in exact:
                historical.append({'mechanism': 'old_fine_wait_helper_plus_three_wait_sites',
                    'source': ref(fine_path), 'shape': row['shape'], 'speedup': row['speedup'],
                    'faster_rounds': row['candidate_faster_rounds'],
                    'own_current_actual_winner': tuple(row['shape']) in own_shapes,
                    'same_selected_traits': True,
                    'scope': 'Exact config direct producer or producer+matching reducer Event for this old candidate; no current policy gain claim.'})
            shared.append({'source': ref(fine_path), 'kind': 'shared_source_mechanism_explanation_only',
                'own_exact_config_performance': False,
                'explanation': 'Global fine_wait helper/site mutation rejected on its finite tested configs. Preserve current synchronization for this config; only exact matching rows above count as own historical performance.'})
        elif config['family'] == 'small_lds_ring':
            shared.append({'source': ref(HERE / 'ring_overlap/results_analysis.json'),
                'kind': 'shared_ring_source_mechanism_explanation_only', 'own_exact_config_performance': False,
                'explanation': '9046/9055 future issue before operand-read experiment rejected for those two types only. Other rings differ in stages/cluster/scale policy; their numerical/identity review is their own evidence and this rejection is not their measured performance.'})
        else:
            shared.append({'source': ref(coverage_path) if coverage else ref(reg_path),
                'kind': 'shared_register_source_mechanism_explanation_only', 'own_exact_config_performance': False,
                'explanation': 'Keep existing queue/waveK/output/cache and scoped reuse policy. Runtime9051 full17 issue-order decision and old global reuse decision do not measure another register config.'})
            for row in register['rows']:
                if row['private_vs_formal']['private_runtime_canonical_traits'] == config['actual_selected_traits']:
                    historical.append({'mechanism': 'old_global_B_scale_reuse_mutation', 'source': ref(reg_path),
                        'shape': row['shape'], 'speedup': row['speedup_ratio_of_medians'],
                        'faster_rounds': row['paired_candidate_faster_rounds'],
                        'own_current_actual_winner': tuple(row['shape']) in own_shapes,
                        'same_selected_traits': True,
                        'scope': 'Own exact old false-reuse config, historical direct-call Event; not a new Oct8 measurement.'})
        if config['actual_configuration_id'] in [9042, 9053, 9054]:
            own_reuse = [row for row in reuse['rows'] if row['actual_configuration_id'] == config['actual_configuration_id']
                         and row['canonical_traits_type'] == config['inventory_baseline_traits']]
            assert len(own_reuse) == config['winner_count']
            for row in own_reuse:
                event = row['verified_prior_event']
                assert tuple(row['shape']) in own_shapes and event['candidate_faster_rounds'] == 5
                historical.append({'mechanism': 'accepted_scoped_runtime_B_scale_reuse', 'source': ref(reuse_path),
                    'shape': row['shape'], 'speedup': event['speedup'], 'faster_rounds': 5,
                    'own_current_actual_winner': True,
                    'scope': 'Own old false baseline to currently selected true alias; selected candidate identity reused through Oct8 formal audit.',
                    'prior_result': {'path': str(old / event['file']), 'sha256': event['sha256']}})
        if config['actual_configuration_id'] in [9071, 9073]:
            assert symbol in n32_base and n32_base[symbol]['instruction_sha256'] == config['instruction_sha256']
            rows = [row for row in n32['rows'] if n32['plan']['targets'][row['target_index']]['kid'] == config['actual_configuration_id']]
            assert len(rows) == 1
            for row in rows:
                event = row['event_confirmation']
                pairs = [{m['label']: m['us_per_call'] for m in event['measurements'] if m['round'] == rnd}
                         for rnd in range(5)]
                historical.append({'mechanism': 'rejected_fixed_N48_to_N32', 'source': ref(n32_path),
                    'own_baseline_device_identity': ref(n32_audit_path), 'shape': row['shape'],
                    'speedup': event['median_speedup']['candidate'],
                    'faster_rounds': sum(pair['candidate'] < pair['baseline'] for pair in pairs),
                    'own_current_actual_winner': tuple(row['shape']) in own_shapes,
                    'same_selected_FUNC_SHA': True,
                    'scope': 'Own exact N48 config old candidate fullEvent; preserveN48. The9071 ATT is not9073 dynamic evidence.'})
        current_event = []
        if symbol in captures:
            target = captures[symbol]['target_index']
            if target == 17:
                current_event = [{'source': ref(HERE / 'fine_startup/results_analysis.json'),
                    'mechanism': 'initial_matrix_before_unchanged_scale_publication',
                    **{k:row[k] for k in ['shape', 'median_speedup', 'faster_rounds']}}
                    for row in finite['fine']['winner_decisions']]
                for row in n64['rows']:
                    historical.append({'mechanism': 'rejected_M80_split2_BN128_to_BN64', 'source': ref(n64_path),
                        'shape': row['shape'], 'speedup': row['speedup'], 'faster_rounds': row['candidate_faster_rounds'],
                        'own_current_actual_winner': tuple(row['shape']) in own_shapes,
                        'scope': 'Own exactfixed9062 old geometry experiment; preserveBN128.'})
            elif target in [14, 15]:
                current_event = [{'source': ref(HERE / 'ring_overlap/results_analysis.json'),
                    'mechanism': 'future_matrix_issue_before_current_operand_LDS_reads',
                    **{k:row[k] for k in ['shape', 'median_speedup', 'faster_rounds']}}
                    for row in finite['ring']['winner_decisions'] if row['shape'] == captures[symbol]['shape']]
            elif target == 20 and coverage:
                current_event = [{'source': ref(coverage_path), 'mechanism': 'same_tile_scale_first_issue',
                    **{k:row[k] for k in ['shape', 'median_speedup', 'faster_rounds']}}
                    for row in coverage['winner_decisions']]
                processing['isolated_register_candidate'].update(status=coverage['status'],
                    full17_coverage_analysis=ref(coverage_path), adopted=False,
                    full17_summary={k:coverage[k] for k in ['full_actual_winner_count','median_positive_winners',
                        'median_regressed_winners','all5rounds_faster_winners','both_order_groups_positive_winners',
                        'uniform_exacttype_decision']})
        assert all(tuple(row['shape']) in own_shapes for row in current_event)
        measured_shapes = {tuple(row['shape']) for row in current_event}
        processing.update(
            own_Oct8_candidate_complete_Event=current_event,
            own_Oct8_candidate_complete_Event_config_measured=bool(current_event),
            own_Oct8_actualwinner_Event_shape_count=len(measured_shapes),
            own_Oct8_all_currentwinner_performance_covered=len(measured_shapes) == config['winner_count'],
            own_historical_exactconfig_Event=historical,
            shared_mechanism_explanations=shared,
            own_exact_local_ATT_measured=symbol in captures,
            finite_round_decision='retain_prior_accepted_runtime_reuse_alias' if config['actual_configuration_id'] in [9042,9053,9054]
                                  else 'reject_finite_candidate_keep_current' if current_event else 'keep_current',
            new_source_or_entry_adopted_this_small_round=False,
            performance_adoption_claim=False,
            evidence_limits='Own numerical gate is correctness evidence. Own finite ATT is localCU shader evidence. Current candidate Event rows and historical exactconfig Event rows are separately listed. Shared mechanisms and otherconfig rejection are not thisconfig performance. Keep is a preservation decision, not a speedup claim.',
            next_action_trigger='Finite small series is closed; no pending candidate, new geometry scope or repeat-until-positive run.')
        if symbol not in captures and config['actual_configuration_id'] not in [9042,9053,9054]:
            processing['status'] = 'keep_current_after_own_exact_config_review_and_numerical_gate'
        if config['actual_configuration_id'] == 9071:
            processing['status'] = 'keep_fixed9071_N48_after_own_ATT_history_and_numerical_review'
        assert symbol in gate_rows
    current_configs = [c for c in review['configurations'] if c['processing']['own_Oct8_candidate_complete_Event_config_measured']]
    review['finite_small_round_closure'] = {'status': 'closed_all_existing_finite_candidates_rejected',
        'current_candidate_Event_exact_config_count': len(current_configs),
        'current_candidate_Event_actualwinner_shape_count': sum(c['processing']['own_Oct8_actualwinner_Event_shape_count'] for c in current_configs),
        'exact_local_ATT_config_count': len(captures), 'all_config_own_numerical_gate_count': len(gate_rows),
        'new_small_source_or_entry_adopted': False, 'pending_small_candidates': 0,
        'fine_startup': ref(HERE / 'fine_startup/results_analysis.json'),
        'ring_overlap': ref(HERE / 'ring_overlap/results_analysis.json'),
        'runtime9051_full17': ref(coverage_path) if coverage else None,
        'prepared9071draft': {'status':'stopped_before_build_no_GPU_no_adoption',
            'closure': ref(HERE / 'fixed9071_issue_order/closure.json'),
            'reason':'Root finite series scope closes existing candidates; ownN48 historical/ATT/numerical evidence supportskeep. No new9071 mechanism claimed measured.'},
        'scope_limit':'4config/21shape current candidate Event coverage; otherconfigs kept using their own selectedsource/identity/numerical and exacthistorical rows where available. No33config performance claim.'}
    for opportunity in review['opportunities']:
        opportunity['finite_round_status'] = 'closed_existing_finite_candidates_keep_current_no_new_mechanism_pending'
        opportunity['preparation_questions_are_performance_evidence'] = False
    for followup in review['conditional_followups']:
        followup['finite_round_status'] = ('selected_and_closed_own_ATT_plus_full17_Event_reject'
            if followup['id'] == 'register9051_external' else 'preparation_control_not_selected_finite_round_closed')
        followup['additional_GPU_pending'] = False


def main():
    path = HERE / 'small_family_review.json'
    review = json.loads(path.read_text())
    attpath = HERE / 'small_att_analysis.json'
    att = json.loads(attpath.read_text())
    assert att['status'] == 'passed_cpu_exact_small_family_ATT_analysis' and att['capture_count'] == 7
    captures = {c['symbol']: c for c in att['captures']}
    findings = {f['target_index']: f for f in att['mechanism_findings']}
    gate_path=HERE/'small_config_gate_audit.json'
    gate=json.loads(gate_path.read_text()) if gate_path.exists() else None
    if gate:assert gate['status']=='passed_all33_small_actual_config_numerical_gate'
    gate_rows={r['symbol']:r for r in gate['rows']} if gate else {}
    before = [(c['symbol'], c['winner_count'], len(c['winner_shapes'])) for c in review['configurations']]
    for config in review['configurations']:
        if config['symbol'] in gate_rows:
            row=gate_rows[config['symbol']]
            config['processing']['own_actual_config_numerical_gate']={
                'audit':ref(gate_path),'target_index':row['target_index'],'shape':row['shape'],
                'signed2repeat_reference_output_workspace_guards_passed':True,
                'performance_measured':False,'authoritative_selected_symbol':row['symbol']}
            if config['processing']['status']=='reviewed_deferred_no_new_GPU_by_this_review':
                config['processing'].update(status='keep_current_after_own_exact_config_review_and_numerical_gate',
                    keep_reason='Own actualselected type/sourcepolicy, exactformal machineidentity and one actualwinner signednumerical/repeat/guards pass. Currentconfig has no directly measured localPC evidence supporting a distinct safe change; preserve its selectedpolicy. Otherconfig rejection is not its performance measurement.',
                    next_action_trigger='A distinct exactconfig request/dependency/address/output mechanism with its ownISA evidence; no automatic crossconfig trait/geometry expansion.',
                    performance_adoption_claim=False)
        if config['symbol'] in captures:
            capture = captures[config['symbol']]
            finding = findings[capture['target_index']]
            config['processing'].update(status=finding['status'],
                exact_local_ATT={'result': ref(attpath), 'target_index': capture['target_index'],
                                 'shape': capture['shape'], 'captured_complete_waves': capture['captured_complete_waves'],
                                 'target_CU': capture['target_CU'], 'instruction_SHA_equal': True,
                                 'separate_signed2repeat_reference_guards_passed': True,
                                 'whole_GPU_performance_measured': False},
                next_action_trigger=finding.get('finite_followup', finding.get('candidate', {}).get('risk_to_resolve')))
            config['processing']['evidence'] = [e for e in config['processing']['evidence'] if e.get('path') != str(attpath)] + [
                {'path': str(attpath), 'sha256': sha(attpath), 'target_index': capture['target_index'],
                 'scope': 'Exact selected FUNC/fullmetadata/descriptor and owncleanowner; one finite CU shader mechanism, no performance adoption.'}]
            if capture['target_index'] == 17:
                config['processing']['isolated_candidate'] = {
                    'status': 'CPU_exact_scope_ISA_passed_pending_GPU',
                    'directory': str(HERE / 'fine_startup'),
                    'source_diff': ref(HERE / 'fine_startup/candidate.diff'),
                    'CPU_device_audit': ref(HERE / 'fine_startup/device_audit.json'),
                    'ISA_review': ref(HERE / 'fine_startup/isa_review.json'),
                    'screen_plan': ref(HERE / 'fine_startup/screen_plan.json'),
                    'winner_event_plan': ref(HERE / 'fine_startup/winner_event_plan.json'),
                    'unselected_entries_equal': 39,
                    'adopted': False}
                decision_path = HERE / 'fine_startup/results_analysis.json'
                if decision_path.exists():
                    decision=json.loads(decision_path.read_text())
                    assert decision['status']=='rejected_no_positive_complete_Event_gain'
                    config['processing']['isolated_candidate'].update(status=decision['status'],
                        results_analysis=ref(decision_path),
                        winner_decisions=[{k:r[k] for k in ['shape','median_us','median_speedup','faster_rounds']}
                                          for r in decision['winner_decisions']],
                        retry_policy=decision['retry_policy'])
            if capture['target_index'] in [14,15] and (HERE/'ring_overlap/isa_review.json').exists():
                config['processing']['isolated_ring_candidate']={
                    'status':'CPU_exact_ring_issue_read_order_passed_pending_GPU',
                    'directory':str(HERE/'ring_overlap'),
                    'source_diff':ref(HERE/'ring_overlap/candidate.diff'),
                    'device_audit':ref(HERE/'ring_overlap/device_audit.json'),
                    'ISA_review':ref(HERE/'ring_overlap/isa_review.json'),
                    'screen_plan':ref(HERE/'ring_overlap/screen_plan.json'),
                    'representative_event_plan':ref(HERE/'ring_overlap/representative_event_plan.json'),
                    '38unselectedentries_equal':True,'adopted':False}
                decision_path=HERE/'ring_overlap/results_analysis.json'
                if decision_path.exists():
                    decision=json.loads(decision_path.read_text())
                    assert decision['status']=='rejected_both_ring_types_no_positive_complete_Event_gain'
                    chosen=[r for r in decision['winner_decisions'] if r['shape']==capture['shape']]
                    assert len(chosen)==1
                    config['processing']['isolated_ring_candidate'].update(status='rejected_for_this_exact_type',
                        results_analysis=ref(decision_path),
                        representative_Event_decision={k:chosen[0][k] for k in ['shape','median_us','median_speedup','faster_rounds']},
                        retry_policy=decision['retry_policy'])
            if capture['target_index']==20 and (HERE/'register_issue_order/isa_review.json').exists():
                config['processing']['isolated_register_candidate']={
                    'status':'CPU_exact_runtime9051_scale_first_passed_pending_GPU',
                    'directory':str(HERE/'register_issue_order'),
                    'source_diff':ref(HERE/'register_issue_order/candidate.diff'),
                    'device_audit':ref(HERE/'register_issue_order/device_audit.json'),
                    'ISA_review':ref(HERE/'register_issue_order/isa_review.json'),
                    'screen_plan':ref(HERE/'register_issue_order/screen_plan.json'),
                    'representative_event_plan':ref(HERE/'register_issue_order/representative_event_plan.json'),
                    '39unselectedentries_equal':True,'adopted':False}
                decision_path=HERE/'register_issue_order/results_analysis.json'
                if decision_path.exists():
                    decision=json.loads(decision_path.read_text())
                    assert decision['status']=='positive_two_representative_Event_requires17winner_coverage'
                    config['processing']['isolated_register_candidate'].update(
                        status='two_positive_representative_Event_pending17actualwinner_coverage',
                        results_analysis=ref(decision_path),
                        representative_Event_decisions=[{k:r[k] for k in ['shape','median_us','median_speedup','faster_rounds']}
                                                        for r in decision['winner_decisions']],
                        coverage_Event_plan=ref(HERE/'register_issue_order/coverage_event_plan.json'))
    finalize_evidence_scope(review, captures, gate_rows)
    assert before == [(c['symbol'], c['winner_count'], len(c['winner_shapes'])) for c in review['configurations']]
    assert len(review['configurations']) == 33 and sum(c['winner_count'] for c in review['configurations']) == 278
    status = {c['symbol']: c['processing']['status'] for c in review['configurations']}
    for parent in review['public_parent_processing']:
        for config in parent['actual_config_entries']:
            config['status'] = status[config['symbol']]
            own = next(c for c in review['configurations'] if c['symbol'] == config['symbol'])
            processing = own['processing']
            config.update(finite_round_decision=processing['finite_round_decision'],
                own_numerical_gate_passed=True, own_local_ATT_measured=processing['own_exact_local_ATT_measured'],
                own_Oct8_candidate_Event_shape_count=processing['own_Oct8_actualwinner_Event_shape_count'],
                own_Oct8_all_currentwinner_performance_covered=processing['own_Oct8_all_currentwinner_performance_covered'],
                own_historical_exactconfig_Event_row_count=len(processing['own_historical_exactconfig_Event']),
                shared_mechanisms_are_performance_evidence=False)
        parent.update(status='closed_all_actual_configs_traceable_keep_or_reject_currentcandidate',
            new_small_source_or_entry_adopted=False,
            own_Oct8_candidate_Event_shape_count=sum(c['own_Oct8_candidate_Event_shape_count'] for c in parent['actual_config_entries']),
            own_numerical_configs_passed=len(parent['actual_config_entries']),
            limits='Configrows identify owncurrentEvent, ownlocalATT, ownhistoricalEvent andsharedmechanism separately; keptconfigs were notall performancebenchmarked.')
    closure = review['finite_small_round_closure']
    review['summary'].update(
        finite_small_round_closed=True, pending_small_candidates=0,
        own_current_candidate_Event_configs=closure['current_candidate_Event_exact_config_count'],
        own_current_candidate_Event_actualwinner_shapes=closure['current_candidate_Event_actualwinner_shape_count'],
        own_exact_local_ATT_configs=7, own_config_numerical_gates=33,
        new_small_source_or_entry_adopted=False, full33config_performance_coverage=False)
    review['root_execution_update'].update(status='seven_valid_exact_ATT_analyzed_one_original_null_excluded',
        valid_capture_count=7, analyzed_result=ref(attpath),
        original9055CU0null_excluded=True, replacement9055CU1independent_claim_passed=True,
        runtime9051crossWG_barrier_spreads_excluded_from_intraWG_conclusions=True,
        finite_small_candidates_closed=True, runtime9051_full17_uniform_rejected=True,
        source_or_entry_adoption_in_this_small_round=False)
    if gate:
        review['all33_config_own_numerical_gate']={'audit':ref(gate_path),'configs':33,
            'signed2repeat_reference_output_workspace_guards_passed':True,
            'full_performance_coverage':False,'descriptive_trait_label_corrections':gate['descriptive_trait_label_corrections']}
    review['generated_utc'] = datetime.now(timezone.utc).isoformat()
    review['status'] = 'CPU_closed33_config18parent_review7exactATT_all_finite_candidates_rejected'
    path.write_text(json.dumps(review, indent=2) + '\n')
    mdpath = HERE / 'small_family_review.md'
    text = mdpath.read_text().split('\n<!-- OCT8_VALID_ATT_UPDATE -->', 1)[0]
    text = text.replace('# 9040 之后的小矩阵与 fine 系列诊断建议', '# 9040 之后的小矩阵与 fine 系列有限优化审查')
    prefix_start = text.index('基线为 Oct8 已应用 selected。')
    prefix_end = text.index('\n\n278 个历史实际赢家', prefix_start)
    text = text[:prefix_start] + ('本轮已收口：fixed9062 initial handoff、9046/9055 ring issue次序和runtime9051 scale-first候选均拒绝，small原selected保留。33个实际config各自数值门、7个自身ATT和4config/21个actualwinner candidate Event已审查；18个public parent可追溯，未测config的keep不声明性能增益。没有新增small source/deviceentry采用，也没有待执行候选。最终分项结果及证据范围见文末。\n\n'
        '基线为 Oct8 已应用 selected。9021本轮已完成；9042/9053/9054三个runtime B-scale复用alias保持已采用状态。本文件由CPU后处理生成；所有GPU实验由root执行。以下原始诊断建议和控制点保留为准备时点记录，当前执行和收口状态见JSON及文末。') + text[prefix_end:]
    text = text.replace('其余配置均明确 reviewed/deferred，不把共享 family 的代表视作其性能实测。',
                        '所有配置现均有明确keep或候选reject收口决定，不把共享family代表视作其性能实测。')
    text += """
<!-- OCT8_VALID_ATT_UPDATE -->

本轮有限small系列已收口：fine_startup、ring_overlap两个exacttype和runtime9051 scale-first均拒绝，原selected保留。没有新增small source或deviceentry采用，没有待跑small候选。已准备但未build的9071草稿停止；N48/Q4/OUTPUT4/falseBscaleReuse保持原策略，自己的旧N32 Event、局部ATT和数值门分别列为证据。

七个有限代表的精确ATT后处理见 [small_att_analysis.json](small_att_analysis.json)。9055原CU0 `code=null`/零wave排除，独立CU1补采的四个完整wave与正式symbol/FUNC/fullmetadata/descriptor、owner/物理卡、stats/ISA吻合。9051的32个完整wave跨多个WG，同PC/occurrence聚合barrier跨度不能推断WG内到达差。所有shader clocks只用于该CU局部机制，不组成全GPU墙钟百分比。

| 本轮候选 | 自身完整Event范围 | 结果 | 决定 |
| --- | --- | --- | --- |
| fixed9062 initial matrix-before-scale | M144/M160两个actualwinner，含split2 matching reducer | speedup0.996596/0.998854，0/5和1/5更快 | 拒绝，保留原initial handoff |
| 9046 steady ring future issue-before-read | [256,7168,16384]一个actualwinner | speedup0.980340，0/5更快 | 拒绝该exacttype候选，保留原ring |
| 9055 steady ring future issue-before-read | [64,7168,7168]一个actualwinner | speedup0.991871，0/5更快 | 拒绝该exacttype候选，保留原ring |
| runtime9051 same-tile scale-first issue | 全部17个冻结actualwinner | 8项median正、8项退、1项平；仅3/17全部5轮更快、7/17两个order组正 | 全exacttype统一拒绝，保留原issue顺序 |

依据分别为 [fine_startup/results_analysis.json](fine_startup/results_analysis.json)、[ring_overlap/results_analysis.json](ring_overlap/results_analysis.json) 和 [register_issue_order/coverage_analysis.json](register_issue_order/coverage_analysis.json)。各候选40entry基线精确匹配Oct8；fine/register候选39未选entry不变，ring候选38未选entry不变。全部自身signed8repeat/reference/guard及每shape51共享池5ABBA完整Event后的重复/reference/guard通过。9051早期两个正代表保留为screen证据，但全17覆盖的mixed结果决定统一拒绝；没有按shape新增阈值或重复coverage寻找正结果。

[small_config_gate_audit.json](small_config_gate_audit.json) 验证33/33 small实际config各自一个actualwinner的signed2repeat/reference/output/workspaceguard，正式symbol和指令SHA均匹配；all44rootgate ownclaim干净。sealedplan中9042/9053/9054的descriptive trait label旧false已在audit校正，实际mangledsymbol和hash都指向当前true alias，执行身份正确。已采用三runtimealias各自六个历史actualwinner的正Event和Oct8selected身份复用仍保留；fixed9070/9072原true策略属于自身source/identity事实，没有冒称本轮新采用性能。

33config/18parent/278历史winner完整映射保留在 [small_family_review.json](small_family_review.json)。每config现在分开记录自身Oct8 candidate Event形状、自身局部ATT、自身历史exactconfig Event、共享机制解释和自身数值门；每parent映射相同字段。当前candidate性能实测为4config/21个actualwinner形状，局部ATT为7config，其余keep依靠各自selectedtype/source/identity/numerical与存在时的自身历史实验。共享ring/register/fine机制解释以及别的config的reject都明确不算该config性能；没有33config全性能实测声明。

| public parent | actual ID / selected fixedK | 自身本轮candidate Event形状数 | 自身ATT配置数 | 自身历史exactconfig Event行数 | 决定 |
| --- | --- | ---: | ---: | ---: | --- |
"""
    for parent in review['public_parent_processing']:
        configs = [next(c for c in review['configurations'] if c['symbol'] == entry['symbol'])
                   for entry in parent['actual_config_entries']]
        labels = [str(c['actual_configuration_id']) + '/K' + c['actual_selected_traits'].split(',')[-1].rstrip('> ').strip()
                  if c['family'] == 'fine_lds_split' else str(c['actual_configuration_id']) for c in configs]
        decisions = sorted({c['processing']['finite_round_decision'] for c in configs})
        text += f"| {parent['parent_id']} | {', '.join(labels)} | {parent['own_Oct8_candidate_Event_shape_count']} | {sum(c['processing']['own_exact_local_ATT_measured'] for c in configs)} | {sum(len(c['processing']['own_historical_exactconfig_Event']) for c in configs)} | {', '.join(decisions)} |\n"
    mdpath.write_text(text)
    print(json.dumps({'status': review['status'], 'configs': 33, 'winners': 278,
                      'parents': len(review['public_parent_processing']), 'valid_ATT': 7}))


if __name__ == '__main__':
    main()
