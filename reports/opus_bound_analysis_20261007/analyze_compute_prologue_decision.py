#!/usr/bin/env python3
"""CPU performance summary of validated prologue winners and seven followups.

Consumes the strict winner coverage snapshot (refresh with --summary-only),
retains paired raw values, and audits the finite fallback Event results. This
script writes one JSON summary and never modifies GPU plans or queues. Signal
thresholds describe results; they do not automatically adopt or reject code.
"""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from statistics import median
import sys

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def require(value, message):
    if not value:
        raise ValueError(message)


def logs(path):
    result = []
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            result.append(row)
    return result


def geometry(kid, shape):
    m, n, k = shape
    block_m = 128 if kid == 9021 else 160
    return {'block_m': block_m, 'block_n': 128,
            'producer_workgroups': ((m + block_m - 1) // block_m) * ((n + 127) // 128),
            'm_tail': bool(m % block_m), 'k128_tiles': k // 128}


def describe(record):
    pairs = record['paired_rounds']
    require(len(pairs) == 5 and [p['round'] for p in pairs] == list(range(5)), 'paired rounds missing')
    b = [p['baseline_us'] for p in pairs]
    c = [p['candidate_us'] for p in pairs]
    change = [p['change_percent'] for p in pairs]
    mids = record['event_median_us']
    require(mids == {'baseline': median(b), 'candidate': median(c)}, 'record median inconsistency')
    slowdown = [100 * (cv / bv - 1) for bv, cv in zip(b, c)]
    return {'parent_id': record['parent_id'], 'shape': record['shape'],
            'geometry': geometry(record['parent_id'], record['shape']),
            'result': record['result'], 'result_sha256': record['result_sha256'],
            'gpu': record['gpu'], 'claim_start': record['claim_epoch']['start']['time'],
            'claim_end': record['claim_epoch']['end']['time'], 'event_median_us': mids,
            'baseline_over_candidate_speedup': mids['baseline'] / mids['candidate'],
            'ratio_of_medians_speedup_percent': 100 * (mids['baseline'] / mids['candidate'] - 1),
            'ratio_of_medians_candidate_time_change_percent': 100 * (mids['candidate'] / mids['baseline'] - 1),
            'paired_rounds': pairs, 'paired_candidate_time_change_percent': slowdown,
            'paired_positive_rounds': sum(v > 0 for v in change),
            'paired_negative_rounds': sum(v < 0 for v in change),
            'five_of_five_faster': all(v > 0 for v in change),
            'five_of_five_slower': all(v < 0 for v in change),
            'minimum_paired_candidate_time_increase_percent': min(slowdown),
            'maximum_paired_candidate_time_increase_percent': max(slowdown),
            'median_paired_speedup_percent': median(change),
            'baseline_range_over_median_percent': 100 * (max(b) - min(b)) / mids['baseline'],
            'candidate_range_over_median_percent': 100 * (max(c) - min(c)) / mids['candidate'],
            'candidate_min_above_baseline_max': min(c) > max(b),
            'candidate_max_below_baseline_min': max(c) < min(b),
            'automatic_rejection': False}


def group(rows):
    if not rows:
        return {'shape_count': 0}
    b = sum(r['event_median_us']['baseline'] for r in rows)
    c = sum(r['event_median_us']['candidate'] for r in rows)
    ratios = [r['baseline_over_candidate_speedup'] for r in rows]
    gm = math.exp(sum(math.log(v) for v in ratios) / len(ratios))
    rounds = []
    for rnd in range(5):
        bm = sum(r['paired_rounds'][rnd]['baseline_us'] for r in rows)
        cm = sum(r['paired_rounds'][rnd]['candidate_us'] for r in rows)
        g = math.exp(sum(math.log(r['paired_rounds'][rnd]['baseline_us'] / r['paired_rounds'][rnd]['candidate_us'])
                        for r in rows) / len(rows))
        rounds.append({'round': rnd, 'baseline_sum_us': bm, 'candidate_sum_us': cm,
                       'sum_time_speedup': bm / cm, 'geomean_shape_speedup': g})
    return {'shape_count': len(rows), 'geomean_shape_speedup': gm, 'geomean_shape_speedup_percent': 100 * (gm - 1),
            'baseline_sum_of_shape_medians_us': b, 'candidate_sum_of_shape_medians_us': c,
            'sum_time_speedup': b / c, 'sum_time_speedup_percent': 100 * (b / c - 1),
            'median_shape_speedup_percent': median(100 * (v - 1) for v in ratios),
            'minimum_shape_speedup_percent': 100 * (min(ratios) - 1),
            'maximum_shape_speedup_percent': 100 * (max(ratios) - 1),
            'ratio_of_medians_faster_shapes': sum(v > 1 for v in ratios),
            'ratio_of_medians_slower_shapes': sum(v < 1 for v in ratios),
            'five_of_five_faster_shapes': sum(r['five_of_five_faster'] for r in rows),
            'five_of_five_slower_shapes': sum(r['five_of_five_slower'] for r in rows),
            'mixed_or_equal_round_direction_shapes': sum(not r['five_of_five_faster'] and not r['five_of_five_slower'] for r in rows),
            'paired_aggregate_rounds': rounds,
            'five_of_five_sum_time_faster': all(r['sum_time_speedup'] > 1 for r in rounds),
            'five_of_five_geomean_faster': all(r['geomean_shape_speedup'] > 1 for r in rounds),
            'five_of_five_candidate_time_increase_sensitivity_counts': {
                str(t): sum(r['minimum_paired_candidate_time_increase_percent'] >= t for r in rows)
                for t in [0.5, 1, 2, 5]},
            'aggregation_meaning': 'One execution per distinct shape; each shape weighted equally. Sum of medians is a synthetic workload total, not an observed combined batch.'}


def audit_fallback(command, expected_libraries, claims, source_head):
    argv = command['argv']
    path = Path(argv[argv.index('--output') + 1])
    result = read(path)
    require(result['status'] == 'passed' and not result.get('contamination'), 'incomplete or contaminated fallback result')
    require(result['source_head'] == source_head and result['libraries'] == expected_libraries,
            'fallback source/library identity differs')
    require(result['event_confirmation_requested'] and not result['profiling_only'] and result['iters'] == 51
            and result['rounds'] == 1 and result['finished'] > result['started'], 'fallback Event protocol differs')
    plan_path = Path(argv[argv.index('--plan') + 1])
    require(result['plan'] == read(plan_path) and result['plan_sha256'] == sha(plan_path) and not result['plan']['workspace'],
            'fallback plan differs')
    require(result['runner_sha256'] == sha(OUT / 'experiment_runner.py'), 'fallback runner changed')
    require(len(result['rows']) == 1, 'fallback command must run one target')
    row = result['rows'][0]
    target = result['plan']['targets'][int(argv[argv.index('--target-index') + 1])]
    require(row['target_index'] == int(argv[argv.index('--target-index') + 1]) and row['kid'] == target['kid']
            and row['shape'] == target['shape'] and row['split'] == 1 and target['signed'] is True and target['seed'] == 17,
            'fallback actual target differs')
    for label in ['baseline', 'candidate']:
        check = row['correctness'][label]
        require(check['repetitions'] == 8 and check['errRatio'] == 0 and check['repeatable']
                and check['output_guards'] and check['workspace_guards'], 'fallback signed8 numerical check missing')
    start = max((c for c in claims if c.get('event') == 'start' and c.get('command', {}).get('name') == command['name']
                 and c['time'] <= result['started']), key=lambda c: c['time'])
    end = min((c for c in claims if c.get('event') == 'end' and c.get('name') == command['name']
               and c['time'] >= result['finished']), key=lambda c: c['time'])
    claim = max((c for c in claims if c.get('event') == 'claimed' and c['time'] <= start['time']), key=lambda c: c['time'])
    require(start['command'] == command and end['returncode'] == 0 and end['contamination'] is False, 'fallback command epoch differs')
    require(result['gpu']['pci_bdf'] == result['gpu']['expected_pci_bdf'] == claim['gpu']['bdf']
            and result['gpu']['HIP_VISIBLE_DEVICES'] == str(claim['gpu']['hip_index']), 'fallback physical GPU mismatch')
    require('--event-confirm' in argv and argv[argv.index('--iters') + 1] == '51'
            and argv[argv.index('--repetitions') + 1] == '8', 'claimed fallback protocol differs')
    event = row['event_confirmation']
    require(event['status'] == 'passed' and event['rounds'] == 5 and event['iters_per_graph'] == 51 and event['shared_pool'],
            'fallback five-round shared Event missing')
    count = event['rotation']['count']
    require(count >= 1 and len(event['pool_pointers']) == count and all(p['workspace'] == 0 for p in event['pool_pointers']),
            'fallback shared pool differs')
    require(len(event['measurements']) == 10, 'fallback ten Event samples missing')
    expected_log, pairs, times = [], [], {'baseline': [], 'candidate': []}
    for rnd in range(5):
        measurements = [m for m in event['measurements'] if m['round'] == rnd]
        order = ['baseline', 'candidate'] if rnd % 2 == 0 else ['candidate', 'baseline']
        require([m['label'] for m in measurements] == order, 'fallback AB/BA order differs')
        values = {}
        for m in measurements:
            require(m['order'] == order and m['iters'] == 51 and m['rotation_count'] == count
                    and m['us_per_call'] == m['event_total_ms'] * 1000 / 51 and m['us_per_call'] > 0
                    and math.isfinite(m['us_per_call']), 'fallback timing unit/protocol differs')
            require([c['pool_index'] for c in m['all_pool_checks']] == list(range(count)), 'fallback pool checks missing')
            for c in m['all_pool_checks']:
                require(c['errRatio'] == 0 and c['output_repeatable'] and c['output_guards'] and c['workspace_guards'],
                        'fallback pool numerical/repeatability/guard failure')
            expected_log.append({'method': 'graph_event_confirmation', 'kid': row['kid'], 'shape': row['shape'],
                                 **{k: m[k] for k in ['round', 'label', 'event_total_ms', 'iters', 'us_per_call', 'rotation_count']}})
            values[m['label']] = m['us_per_call']
            times[m['label']].append(m['us_per_call'])
        pairs.append({'round': rnd, 'order': order, 'baseline_us': values['baseline'], 'candidate_us': values['candidate'],
                      'change_percent': 100 * (values['baseline'] / values['candidate'] - 1)})
    actual_log = [v for v in logs(Path(command['log'])) if v.get('method') == 'graph_event_confirmation']
    require(actual_log == expected_log, 'fallback ten Event log samples differ')
    mids = {label: median(v) for label, v in times.items()}
    require(mids == event['median_us'] and event['median_speedup']['candidate'] == mids['baseline'] / mids['candidate'],
            'fallback median/ratio mismatch')
    record = {'parent_id': row['kid'], 'shape': row['shape'], 'result': str(path.relative_to(ROOT)),
              'result_sha256': sha(path), 'log_sha256': sha(Path(command['log'])),
              'gpu': result['gpu'], 'claim_epoch': {'claim': claim, 'start': start, 'end': end},
              'plan_sha256': result['plan_sha256'], 'runner_sha256': result['runner_sha256'],
              'event_median_us': mids, 'paired_rounds': pairs, 'shared_pool_checks_per_measurement': count,
              'numerical_guards_and_exact_logs_passed': True}
    return record


def main():
    coverage_path = OUT / 'results/compute_prologue_winner_event_coverage.json'
    coverage = read(coverage_path)
    require(coverage['status'] in ['partial_201_Event_coverage', 'complete_201_Event_coverage'], 'strict coverage status differs')
    require(coverage['remaining_queue_written'] is False, 'refresh coverage with --summary-only during GPU execution')
    require(coverage['summary']['expected_winner_shapes'] == 201, 'winner domain differs')
    for label, lib in coverage['libraries'].items():
        require(sha(Path(lib['path'])) == lib['sha256'], 'current prologue libraries changed')
    winners = []
    for row in coverage['covered_winners']:
        for record in row['clean_records']:
            require(sha(ROOT / record['result']) == record['result_sha256'], 'validated winner result changed')
        summary = describe(row['summary_record'])
        summary['clean_window_count'] = len(row['clean_records'])
        summary['all_clean_windows'] = [describe(r) for r in row['clean_records']]
        summary['all_clean_windows_five_of_five_slower'] = all(r['five_of_five_slower'] for r in summary['all_clean_windows'])
        summary['all_clean_windows_five_of_five_faster'] = all(r['five_of_five_faster'] for r in summary['all_clean_windows'])
        summary['independent_clean_window_repeat_available'] = len(row['clean_records']) > 1
        winners.append(summary)
    expected_parent = {9021: 42, 9022: 159}
    parent_groups = {str(kid): {**group([r for r in winners if r['parent_id'] == kid]),
                              'expected_winner_shapes': expected, 'coverage_complete': sum(r['parent_id'] == kid for r in winners) == expected}
                     for kid, expected in expected_parent.items()}
    inventory = read(OUT / 'shape_inventory.json')
    expected_k = Counter((r['parent_id'], r['K']) for r in inventory['rows'] if r['parent_id'] in expected_parent)
    require(sum(expected_k.values()) == 201, 'current inventory winner count differs')
    k_groups = {}
    for kid in expected_parent:
        k_groups[str(kid)] = {}
        for k in sorted(k for p, k in expected_k if p == kid):
            selected = [r for r in winners if r['parent_id'] == kid and r['shape'][2] == k]
            k_groups[str(kid)][str(k)] = {**group(selected), 'expected_winner_shapes': expected_k[(kid, k)],
                                        'remaining_winner_shapes': expected_k[(kid, k)] - len(selected),
                                        'coverage_complete': len(selected) == expected_k[(kid, k)]}
    queue_path = OUT / 'compute_support_event_followups_queue.json'
    queue = read(queue_path)
    require(len(queue['commands']) == 7, 'finite fallback followup count differs')
    claims = logs(OUT / 'gpu_claim_log.jsonl')
    fallback_records, incomplete = [], []
    for command in queue['commands']:
        try:
            record = audit_fallback(command, coverage['libraries'], claims, coverage['source_head'])
            fallback_records.append(record)
        except (OSError, ValueError, KeyError, TypeError) as error:
            incomplete.append({'command': command['name'], 'reason': str(error)})
    fallback = [describe(r) for r in fallback_records]
    five_slower = [r for r in winners if r['five_of_five_slower']]
    report = {'status': 'cpu_performance_summary_pending_human_adoption_decision',
              'new_gpu_execution': False, 'adopted': False, 'plans_or_queues_modified': False,
              'strict_winner_coverage_sha256': sha(coverage_path), 'source_head': coverage['source_head'],
              'libraries': coverage['libraries'], 'summary': coverage['summary'],
              'aggregation_policy': 'Deduplicate parent+shape. Use the latest complete clean Event window once per shape, retaining every accepted prior window. Ratios >1 mean faster candidate.',
              'all_covered_winners': group(winners), 'parents': parent_groups, 'parent_K_groups': k_groups,
              'winners': winners, 'five_of_five_slower_winner_signals': five_slower,
              'finite_fallback_followups': {'expected': 7, 'strict_clean_complete': len(fallback),
                    'coverage_complete': len(fallback) == 7, 'group': group(fallback),
                    'rows': fallback, 'strict_evidence': fallback_records, 'not_accepted': incomplete},
              'interpretation': ['No automatic adoption or rejection. Every five-of-five negative direction is listed with all paired values and measured ranges.',
                                 '0.5/1/2/5% paired slowdown counts are sensitivity summaries, not acceptance thresholds.',
                                 'A single negative median or five small negative differences can reflect drift/noise; ranges and independent repeats remain visible. Do not rerun until positive.',
                                 'Parent and K summaries from partial coverage do not predict the remaining shapes. Final decision uses the complete current-winner domain and finite observed fallback signals.',
                                 'Event rounds share a physical claim and input pool; five rounds are not five independent GPU windows.',
                                 'Per-round aggregates combine independent shape executions and different GPU claims. Synthetic totals are descriptive, not a measured batch.',
                                 'Check-only formal API results establish numerical/loaded-module identity separately, not optimization speed.']}
    require('torch' not in sys.modules, 'CPU summary imported torch')
    output = OUT / 'results/compute_prologue_performance_decision.json'
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'output': str(output), 'coverage': coverage['summary'], 'parents': parent_groups,
                      'five_of_five_slower_winners': len(five_slower), 'fallback_clean': len(fallback),
                      'fallback_not_accepted': incomplete}))


if __name__ == '__main__':
    main()
