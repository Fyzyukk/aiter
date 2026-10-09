#!/usr/bin/env python3
"""Audit the once full17 exactruntime9051 coverage and report uniform evidence."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
spec = importlib.util.spec_from_file_location('register_coverage_claim_cpu', HERE.parent / 'small_att_analysis.py')
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ref(path):
    return {'path': str(path), 'sha256': sha(path)}


def main():
    prep = json.loads((HERE / 'coverage_preparation.json').read_text())
    queue_path, plan_path = HERE / 'coverage_queue.json', HERE / 'coverage_event_plan.json'
    queue = json.loads(queue_path.read_text())
    assert prep['queue'] == ref(queue_path) and prep['plan'] == ref(plan_path)
    assert len(queue['commands']) == 1 and queue['root_only_GPU_execution']
    command = queue['commands'][0]
    argv = command['argv']
    result_path = Path(argv[argv.index('--output') + 1])
    result = json.loads(result_path.read_text())
    plan = json.loads(plan_path.read_text())
    assert result['status'] == 'passed' and result['plan'] == plan
    assert result['plan_sha256'] == sha(plan_path)
    assert result['runner_sha256'] == prep['runner']['sha256'] == sha(prep['runner']['path'])
    assert result['libraries'] == plan['sealed_libraries']
    assert result['rounds'] == 1 and result['iters'] == 51 and not result['profiling_only']
    assert result['event_confirmation_requested'] is True
    assert plan['workspace'] is False and plan['predeclared_once'] is True
    assert plan['historical_actual_winner_count'] == 17
    for side in ['baseline', 'candidate']:
        assert plan['sealed_libraries'][side] == ref(plan['libraries'][side])
    assert plan['device_audit'] == ref(HERE / 'device_audit.json')
    assert plan['ISA_review'] == ref(HERE / 'isa_review.json')
    inventory = json.loads((HERE.parent / 'small_family_review.json').read_text())
    config = next(c for c in inventory['configurations'] if c['actual_configuration_id'] == 9051)
    expected_shapes = [s['shape'] for s in config['winner_shapes']]
    assert expected_shapes == prep['winner_shapes'] == [t['shape'] for t in plan['targets']]
    assert len(result['rows']) == len(expected_shapes) == 17
    assert plan['selected_symbol'] == config['symbol']
    claims_path = HERE / 'coverage_claim.jsonl'
    claims = [json.loads(line) for line in claims_path.read_text().splitlines() if line.strip()]
    own = common.clean_claim(command, result, claims)
    rows = []
    for index, row in enumerate(result['rows']):
        assert row['target_index'] == index and row['shape'] == expected_shapes[index]
        target = plan['targets'][index]
        assert target['kid'] == 9051 and target['symbol'] == config['symbol']
        assert target['instruction_sha256'] == config['instruction_sha256']
        assert target['signed'] is True and target['seed'] == 17
        assert set(row['correctness']) == {'baseline', 'candidate'}
        assert all(c == {'repetitions': 8, 'errRatio': 0, 'output_guards': True,
                         'workspace_guards': True, 'repeatable': True}
                   for c in row['correctness'].values())
        event = row['event_confirmation']
        assert event['status'] == 'passed' and event['rounds'] == 5 and event['iters_per_graph'] == 51
        assert event['shared_pool'] is True and event['rotation']['count'] == len(event['pool_pointers']) == 51
        assert all(set(p) == {'a', 'b', 'c', 'sfa', 'sfb', 'workspace'}
                   and all(p[n] > 0 and p[n] % 16 == 0 for n in ['a', 'b', 'c', 'sfa'])
                   and p['sfb'] > 0 and p['workspace'] == 0 for p in event['pool_pointers'])
        assert len({p['b'] for p in event['pool_pointers']}) == len({p['c'] for p in event['pool_pointers']}) == 51
        assert event['includes'] == 'complete private launch; split-K producer and reducer'
        measurements = event['measurements']
        assert len(measurements) == 10
        rounds = []
        for rnd in range(5):
            pair = {m['label']: m for m in measurements if m['round'] == rnd}
            assert set(pair) == {'baseline', 'candidate'}
            order = ['baseline', 'candidate'] if rnd % 2 == 0 else ['candidate', 'baseline']
            for measurement in pair.values():
                assert measurement['order'] == order
                assert measurement['iters'] == measurement['rotation_count'] == 51
                assert len(measurement['all_pool_checks']) == 51
                assert all(check == {'pool_index': n, 'output_repeatable': True,
                                      'workspace_repeatable': False, 'errRatio': 0,
                                      'output_guards': True, 'workspace_guards': True}
                           for n, check in enumerate(measurement['all_pool_checks']))
                assert 0 < measurement['us_per_call'] < float('inf')
                assert math.isclose(measurement['us_per_call'], measurement['event_total_ms'] * 1000 / 51,
                                    rel_tol=1e-12, abs_tol=1e-12)
            baseline, candidate = pair['baseline']['us_per_call'], pair['candidate']['us_per_call']
            rounds.append({'round': rnd, 'order': order, 'baseline_us': baseline,
                           'candidate_us': candidate, 'speedup': baseline / candidate,
                           'candidate_time_change_percent': 100 * (candidate / baseline - 1)})
        baseline = statistics.median(r['baseline_us'] for r in rounds)
        candidate = statistics.median(r['candidate_us'] for r in rounds)
        assert math.isclose(event['median_us']['baseline'], baseline, rel_tol=1e-12)
        assert math.isclose(event['median_us']['candidate'], candidate, rel_tol=1e-12)
        assert math.isclose(event['median_speedup']['candidate'], baseline / candidate, rel_tol=1e-12)
        paired = [r['speedup'] for r in rounds]
        order_medians = {name: statistics.median(r['speedup'] for r in rounds if r['order'][0] == first)
                         for name, first in [('AB', 'baseline'), ('BA', 'candidate')]}
        rows.append({'target_index': index, 'shape': row['shape'],
                     'all_signed8repeat_reference_output_guards_passed': True,
                     'all510_shared_pool_output_repeat_reference_guards_passed': True,
                     'workspace_repeatability': 'not_applicable_split1_no_global_workspace',
                     'complete_call_scope': 'One split1 producer launch including original in-WG waveK LDS/FP32 reduction and output; no separate reducer launch.',
                     'independent_timing_median_us': row['median_us'],
                     'complete_shared_pool_ABBA_Event': rounds,
                     'median_us': {'baseline': baseline, 'candidate': candidate},
                     'median_speedup': baseline / candidate,
                     'paired_median_speedup': statistics.median(paired),
                     'median_time_change_percent': 100 * (candidate / baseline - 1),
                     'faster_rounds': sum(s > 1 for s in paired),
                     'order_group_paired_median_speedup': order_medians,
                     'paired_speedup_min': min(paired), 'paired_speedup_max': max(paired)})
    positive = [r for r in rows if r['median_speedup'] > 1]
    regressions = [r for r in rows if r['median_speedup'] < 1]
    all_faster = [r for r in rows if r['faster_rounds'] == 5]
    order_consistent = [r for r in rows if all(v > 1 for v in r['order_group_paired_median_speedup'].values())]
    uniform = len(positive) == len(all_faster) == len(order_consistent) == 17
    status = ('positive_runtime9051_full17_uniform_Event_pending_formalAPI' if uniform
              else 'rejected_runtime9051_full17_mixed_or_weak_Event')
    report = {'status': status, 'audit_status': 'passed_CPU_full17_exact_identity_claim_correctness_Event_review',
              'generated_utc': datetime.now(timezone.utc).isoformat(), 'cpu_postprocessing_only': True,
              'adopted': False, 'production_modified': False, 'exact_configuration_id': 9051,
              'selected_symbol': config['symbol'], 'full_actual_winner_count': 17,
              'median_positive_winners': len(positive), 'median_regressed_winners': len(regressions),
              'all5rounds_faster_winners': len(all_faster), 'both_order_groups_positive_winners': len(order_consistent),
              'uniform_exacttype_decision': 'eligible_for_root_formal_API_gate' if uniform else 'keep_current_selected_reject_this_candidate',
              'reason': ('All17 frozen winners have positive median, all5 paired rounds faster, and positive AB/BA group medians; root formalAPI/integration remains required.' if uniform
                         else 'Full17 immutable exacttype coverage is mixed or weak. Preserve currentselected; two earlier positive representatives cannot override other ownwinner results.'),
              'retry_policy': 'Finite candidate closed; no post-hoc shape subset, threshold or repeat coverage to seek a positive result.',
              'limits': 'All5 paired signs describe this clean local run, no independent p-value/significance claim. One independent screen timing round is retained for transparency and is not the completeEvent adoption gate. Generic runner includes label is broader than actual split1 launch; no global workspace/reducer asserted.',
              'clean_claim': own, 'winner_decisions': rows,
              'plan': ref(plan_path), 'queue': ref(queue_path), 'result': ref(result_path),
              'claim_log': ref(claims_path), 'preparation': ref(HERE / 'coverage_preparation.json'),
              'representative_review': ref(HERE / 'results_analysis.json'), 'script': ref(Path(__file__))}
    (HERE / 'coverage_analysis.json').write_text(json.dumps(report, indent=2) + '\n')
    markdown = (HERE / 'review.md').read_text().split('\n<!-- FULL17_COVERAGE_DECISION -->', 1)[0]
    markdown += '\n<!-- FULL17_COVERAGE_DECISION -->\n\n'
    markdown += f"17个冻结实际winner的完整sharedpool Event审查已完成：{report['uniform_exacttype_decision']}。{len(positive)}/17 median正值、{len(all_faster)}/17全部5轮更快、{len(order_consistent)}/17两个order组median正值；所有signed8repeat/reference/outputguard和每shape510池outputrepeat/reference/guard通过。见 [coverage_analysis.json](coverage_analysis.json)。\n\n"
    markdown += '| shape M,N,K | baseline median us | candidate median us | speedup | 更快轮数 | AB / BA paired medians |\n| --- | ---: | ---: | ---: | ---: | ---: |\n'
    for row in rows:
        markdown += f"| {','.join(map(str, row['shape']))} | {row['median_us']['baseline']:.6f} | {row['median_us']['candidate']:.6f} | {row['median_speedup']:.6f} | {row['faster_rounds']}/5 | {row['order_group_paired_median_speedup']['AB']:.6f} / {row['order_group_paired_median_speedup']['BA']:.6f} |\n"
    markdown += '\n本次以整个exact runtime9051类型统一判定，不按结果新增shape子集或阈值；不重跑以寻找正结果。当前未修改production。split1完整调用包含producer内部waveK LDS/FP32归约与输出，不含单独reducer/globalworkspace。\n'
    (HERE / 'review.md').write_text(markdown)
    print(json.dumps({k: report[k] for k in ['status', 'audit_status', 'median_positive_winners',
                                           'median_regressed_winners', 'all5rounds_faster_winners',
                                           'both_order_groups_positive_winners', 'uniform_exacttype_decision']}))


if __name__ == '__main__':
    main()
