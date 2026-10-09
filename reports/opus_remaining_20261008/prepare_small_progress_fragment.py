#!/usr/bin/env python3
"""Emit a standalone small33 inventory-symbol progress fragment and two-paragraph appendix."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ref(path):
    return {'path': str(Path(path).resolve()), 'sha256': sha(path)}


def main():
    inventory_path = HERE / 'inventory.json'
    review_path = HERE / 'small_family_review.json'
    inventory = json.loads(inventory_path.read_text())
    review = json.loads(review_path.read_text())
    gate_path = HERE / 'small_config_gate_audit.json'
    gate = json.loads(gate_path.read_text())
    att_path = HERE / 'small_att_analysis.json'
    att = json.loads(att_path.read_text())
    closure_path = HERE / 'small_final_closure_audit.json'
    closure = json.loads(closure_path.read_text())
    assert review['status'] == 'CPU_closed33_config18parent_review7exactATT_all_finite_candidates_rejected'
    assert closure['status'] == 'passed_CPU_small33config18parent_final_closure_scope_audit'
    assert closure['review'] == ref(review_path) and closure['ATT'] == ref(att_path)
    assert gate['status'] == 'passed_all33_small_actual_config_numerical_gate'
    assert att['status'] == 'passed_cpu_exact_small_family_ATT_analysis'
    inventory_rows = {row['symbol']: row for row in inventory['entries'] if row['parent_id'] >= 9040}
    configs = {row['symbol']: row for row in review['configurations']}
    assert set(inventory_rows) == set(configs) and len(configs) == 33
    entries = {}
    for symbol, item in inventory_rows.items():
        config = configs[symbol]
        processing = config['processing']
        label_stale = config['actual_selected_traits'] != item['traits']
        if label_stale:
            assert config['actual_configuration_id'] in [9042, 9053, 9054]
            assert item['traits'] == config['inventory_baseline_traits']
        assert config['instruction_sha256'] == item['instruction_sha256']
        assert config['winner_count'] == item['winner_count']
        assert [row['shape'] for row in config['winner_shapes']] == item['winner_shapes']
        assert config['actual_configuration_id'] in item['actual_configuration_ids']
        assert item['parent_id'] in config['public_parent_ids']
        current = processing['own_Oct8_candidate_complete_Event']
        historical = processing['own_historical_exactconfig_Event']
        evidence = [
            {'kind': 'own_selected_source_and_formal_device_identity',
             'source': review['baseline']['formal_identity_audit'],
             'selected_instruction_sha256': config['instruction_sha256'],
             'selected_descriptor_normalized_sha256': config['descriptor_normalized_sha256'],
             'scope': 'Own inventory symbol, selected traits, FUNC/fullmetadata/normalized descriptor; baseline identity, no new performance measurement.'},
            {'kind': 'own_actual_config_numerical_gate',
             **processing['own_actual_config_numerical_gate'],
             'scope': 'Own one actualwinner signed2repeat/reference/output/workspace guards; correctness only.'}]
        if processing['own_exact_local_ATT_measured']:
            evidence.append({'kind': 'own_exact_local_CU_ATT', **processing['exact_local_ATT'],
                'scope': 'Own finite shape/CU/complete waves; no fullGPU wall-time or unmeasured winner performance.'})
        for row in current:
            evidence.append({'kind': 'own_this_round_candidate_complete_Event', **row,
                'scope': 'This config and this actualwinner shape only, clean shared51pool5ABBA full launch, signed8repeat/reference/guards.'})
        for row in historical:
            evidence.append({'kind': 'own_historical_exactconfig_Event', **row})
        for row in processing['shared_mechanism_explanations']:
            evidence.append({'kind': 'shared_mechanism_explanation_no_own_performance', **row})
        entries[symbol] = {
            'inventory_symbol': symbol, 'selected_symbol': config['symbol'],
            'parent_id': item['parent_id'], 'public_parent_ids': config['public_parent_ids'],
            'actual_configuration_ids': item['actual_configuration_ids'],
            'traits': config['actual_selected_traits'], 'inventory_descriptive_traits': item['traits'],
            'inventory_descriptive_traits_corrected': label_stale,
            'trait_label_authority': 'Selected mangledsymbol and instructionSHA; Oct7 pre-alias descriptivefalse label preserved as historical inventory text.',
            'historical_winner_count': item['winner_count'],
            'state': processing['status'], 'current_optimization_state': processing['status'],
            'decision': processing['finite_round_decision'],
            'correctness_gate': 'passed_own_actualconfig_signed2repeat_reference_output_workspace_guards',
            'new_source_or_entry_adopted_this_small_round': False,
            'own_this_round_candidate_performance_measured': bool(current),
            'own_this_round_candidate_Event_actualwinner_shapes': [row['shape'] for row in current],
            'own_this_round_all_actualwinner_performance_covered': processing['own_Oct8_all_currentwinner_performance_covered'],
            'own_exact_local_ATT_measured': processing['own_exact_local_ATT_measured'],
            'own_historical_exactconfig_Event_row_count': len(historical),
            'evidence': evidence, 'review': ref(review_path),
            'evidence_limit': processing['evidence_limits'],
            'pending_action': None,
            'closure_policy': 'Keep own selected policy or reject measured candidate uniformly; no post-hoc scope and no repeat-until-positive run.'}
    parents = {}
    for parent in review['public_parent_processing']:
        children = [entries[row['symbol']] for row in parent['actual_config_entries']]
        decisions = {row['decision'] for row in children}
        if 'reject_finite_candidate_keep_current' in decisions:
            decision = 'keep_current_selected_after_own_measured_candidate_rejection'
        elif 'retain_prior_accepted_runtime_reuse_alias' in decisions:
            decision = 'retain_prior_accepted_runtime_alias_keep_other_own_selected_configs'
        else:
            decision = 'keep_own_current_selected_configs'
        parents[str(parent['parent_id'])] = {
            'parent_id': parent['parent_id'], 'state': parent['status'], 'decision': decision,
            'historical_winner_count': parent['winner_count'], 'actual_config_count': len(children),
            'inventory_symbols': [row['inventory_symbol'] for row in children],
            'actual_configuration_ids': sorted({n for row in children for n in row['actual_configuration_ids']}),
            'own_numerical_configs_passed': len(children),
            'own_exact_local_ATT_config_count': sum(row['own_exact_local_ATT_measured'] for row in children),
            'own_this_round_candidate_Event_config_count': sum(row['own_this_round_candidate_performance_measured'] for row in children),
            'own_this_round_candidate_Event_actualwinner_shape_count': sum(len(row['own_this_round_candidate_Event_actualwinner_shapes']) for row in children),
            'own_this_round_all_parent_actualwinner_performance_covered': all(row['own_this_round_all_actualwinner_performance_covered'] for row in children),
            'own_historical_exactconfig_Event_row_count': sum(row['own_historical_exactconfig_Event_row_count'] for row in children),
            'new_small_source_or_entry_adopted': False,
            'evidence': [ref(review_path), ref(gate_path), ref(closure_path)],
            'evidence_limit': 'Children retain separate own numerical, own localATT, own current candidate Event and own historical Event scopes. Shared mechanisms are not parent performance; a keep decision does not assert allwinner speedup.',
            'pending_action': None}
    assert len(parents) == 18 and sum(row['historical_winner_count'] for row in entries.values()) == 278
    assert sum(row['own_this_round_candidate_performance_measured'] for row in entries.values()) == 4
    assert sum(len(row['own_this_round_candidate_Event_actualwinner_shapes']) for row in entries.values()) == 21
    assert sum(row['own_exact_local_ATT_measured'] for row in entries.values()) == 7
    reducers = {row['name']: {'symbol': row['name'], 'parent_id': row['parent_id'],
        'state': 'keep_current_unchanged_matching_reducer', 'instruction_sha256': row['instruction_sha256'],
        'descriptor_normalized_sha256': row['descriptor_normalized_sha256'],
        'evidence': [review['baseline']['formal_identity_audit']],
        'evidence_limit': 'Five reducer entries are machine-identity/complete-call support. Fixed9062 initial handoff Event included its matching split2 reducer; no separate performance measurement claimed for every reducer.'}
        for row in review['reducer_entries']}
    assert len(reducers) == 5
    output = {'status': 'CPU_prepared_final_small33_inventory_symbol_progress_fragment',
        'generated_utc': datetime.now(timezone.utc).isoformat(), 'cpu_only': True,
        'root_family_progress_modified': False, 'production_modified': False,
        'merge_contract': 'Match root entries by inventory symbol; copy state/evidence/evidence_limit and retain symbol identity. This fragment adds no selected source or entry.',
        'summary': {'public_parents': 18, 'actual_producer_configs_with_historical_winner': 33,
            'actual_configuration_ids': 30, 'historical_actualwinner_shapes': 278,
            'formal_producer_entries': 35, 'formal_reducer_entries': 5,
            'formal_producer_entries_without_historical_winner': 2,
            'own_numerical_configs': 33, 'own_exact_local_ATT_configs': 7,
            'own_this_round_candidate_Event_configs': 4,
            'own_this_round_candidate_Event_actualwinner_shapes': 21,
            'all33_config_performance_measured': False, 'pending_small_candidates': 0,
            'new_small_source_or_entry_adopted': False},
        'entries_by_inventory_symbol': entries, 'parents_by_id': parents,
        'reducers_by_symbol': reducers,
        'evidence': {'inventory': ref(inventory_path), 'review': ref(review_path),
                     'numerical_gate': ref(gate_path), 'ATT': ref(att_path),
                     'closure_audit': ref(closure_path), 'script': ref(Path(__file__))},
        'evidence_limit': 'This closes finite small experiments. It does not claim all278 winners were performance measured; the root three-entry official API/integration remains a separate gate.'}
    fragment_path = HERE / 'small_family_progress_fragment.json'
    fragment_path.write_text(json.dumps(output, indent=2) + '\n')
    paragraph1 = ('small与fine分支本轮按33个有历史winner的实际producer symbol配置收口，映射18个public parent、30个actual ID和278个历史winner。'
        '候选性能实测严格为4个配置、21个actual-winner形状：fixed9062 initial matrix-before-scale覆盖M144/M160两个winner且完整调用包含matching split2 reducer；'
        '9046与9055 steady ring issue-before-read各测一个自身winner；runtime9051 scale-first覆盖其全部17个winner。'
        '四个exact配置的候选均拒绝并保留原selected：fixed9062两项speedup0.996596/0.998854，9046/9055为0.980340/0.991871；'
        '9051完整17项为8项median正、8项退、1项平，仅3项全部5轮更快。所有自身signed8repeat、reference、重复及guard和clean sharedpool完整Event审查通过。'
        '两个9051早期正代表仅是进入固定覆盖的screen依据，全17 mixed结果作统一拒绝，不按shape新增阈值或重跑寻找正结果。'
        '本轮small没有新增source/device entry采用，9042/9053/9054此前已采用runtime B-scale alias保留，9071草稿在build前停止。')
    paragraph2 = ('证据范围分别保存：7个exact配置各有一个自身有限形状的局部CU ATT，33个配置各自一个actualwinner的signed2repeat/reference/output/workspace guard数值门；'
        '它们分别证明局部机制和正确性，均不扩大为全部winner性能覆盖。9055原CU0 code=null/零wave记录排除，CU1补采独立核过身份；'
        '9051的32个ATT wave跨多个WG，聚合同PC barrier跨度不用于WG内到达差。该small正式入口集合为35个producer与5个独立reducer，'
        '其中33个producer与历史winner相交、两个兼容/runtime producer无历史winner；5个reducer保留机器身份，未声称每个reducer有独立性能实测。'
        '各配置的keep依据自身selected source/type、正式FUNC/fullmetadata/descriptor、数值门及存在时的自身历史exact-config Event；'
        '共享ring/register/fine机制说明、静态资源或其他配置reject都不算该配置性能。完整配置记录见 [small审查](../opus_remaining_20261008/small_family_review.json)，'
        '范围核验见 [small最终审计](../opus_remaining_20261008/small_final_closure_audit.json)。')
    markdown = paragraph1 + '\n\n' + paragraph2 + '\n\n'
    markdown += '| public parent | actual配置（同ID按fixedK区分） | 历史winner数 | 自身本轮候选Event形状 | 自身ATT配置 | 自身数值门配置 | 最终状态 |\n'
    markdown += '| --- | --- | ---: | ---: | ---: | ---: | --- |\n'
    labels_by_parent = {}
    for parent_id, parent in parents.items():
        children = [configs[symbol] for symbol in parent['inventory_symbols']]
        labels = [str(row['actual_configuration_id']) + ('/K' + str(row['geometry']['fixed_k'])
                  if row['family'] == 'fine_lds_split' else '') for row in children]
        labels_by_parent[parent_id] = labels
        note = ('保留；拒绝自身有限候选' if 'rejection' in parent['decision']
                else '保留已采用runtime alias；其余配置keep' if 'prior_accepted' in parent['decision']
                else 'keep自身当前配置')
        if int(parent_id) == 9042:
            note += '；9071保留N48'
        markdown += f"| {parent_id} | {', '.join(labels)} | {parent['historical_winner_count']} | {parent['own_this_round_candidate_Event_actualwinner_shape_count']} | {parent['own_exact_local_ATT_config_count']} | {parent['own_numerical_configs_passed']} | {note} |\n"
    markdown += '\n表中0项candidate Event表示该parent本轮保留自身配置，没有冒称该parent全部winner已测或已获得新性能增益；历史exact-config Event另存逐symbol记录。\n'
    appendix_path = HERE / 'small_final_document_append.md'
    appendix_path.write_text(markdown)
    audit = {'status': 'passed_CPU_small_progress_fragment_inventory_symbol_and_document_scope_audit',
        'root_family_progress_modified': False, 'original_Compute_Memory_modified': False,
        'fragment': ref(fragment_path), 'document_appendix': ref(appendix_path),
        'exact_inventory_symbol_keys': 33, 'parents': 18, 'historical_winners': 278,
        'own_current_candidate_Event_configs': 4, 'own_current_candidate_Event_actualwinner_shapes': 21,
        'own_local_ATT_configs': 7, 'own_numerical_configs': 33, 'reducers': 5,
        'allwinner_performance_claim': False, 'root_official_API_status': 'separate_pending_root_gate'}
    (HERE / 'small_progress_fragment_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps(audit))


if __name__ == '__main__':
    main()
