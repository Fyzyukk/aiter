#!/usr/bin/env python3
"""CPU audit of the second prologue window; does not import torch or call HIP."""
import hashlib
import json
import math
from pathlib import Path
from statistics import median

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def log_rows(path):
    rows = []
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


CLAIMS = log_rows(OUT / 'gpu_claim_log.jsonl')
INVENTORY = read(OUT / 'shape_inventory.json')
SHAPES = {(r['M'], r['N'], r['K']): r for r in INVENTORY['rows']}


def claim_for(name, result):
    # A repeated command can have multiple starts and ends. Match this epoch.
    start = max((r for r in CLAIMS if r.get('event') == 'start'
                 and r.get('command', {}).get('name') == name
                 and r['time'] <= result['started']), key=lambda r: r['time'])
    end = min((r for r in CLAIMS if r.get('event') == 'end'
               and r.get('name') == name and r['time'] >= result['finished']),
              key=lambda r: r['time'])
    claim = max((r for r in CLAIMS if r.get('event') == 'claimed'
                 and r['time'] <= start['time']), key=lambda r: r['time'])
    assert end['returncode'] == 0 and end['contamination'] is False
    assert claim['gpu']['bdf'] == result['gpu']['pci_bdf'] == '0000:85:00.0'
    assert result['gpu']['HIP_VISIBLE_DEVICES'] == '5'
    return {'claim': claim, 'start': start, 'end': end}


def check_base(name, expected_iters):
    path = OUT / 'results' / (name + '.json')
    result = read(path)
    assert result['status'] == 'passed' and not result.get('contamination')
    assert result['source_head'] == INVENTORY['source_head']
    assert result['rounds'] == 1 and result['iters'] == expected_iters
    for label, lib in result['libraries'].items():
        assert sha(Path(lib['path'])) == lib['sha256']
    for row in result['rows']:
        target = result['plan']['targets'][row['target_index']]
        assert (row['kid'], row['shape']) == (target['kid'], target['shape'])
        assert target['signed'] is True
        for label in ['baseline', 'candidate']:
            c = row['correctness'][label]
            assert c['repetitions'] == 8 and c['errRatio'] == 0
            assert c['repeatable'] and c['output_guards'] and c['workspace_guards']
        for t in row['timings']:
            assert t['errRatio'] == 0 and t['output_guards'] and t['workspace_guards']
            assert math.isfinite(t['us']) and t['us'] > 0
    log_path = path.with_suffix('.log')
    logs = log_rows(log_path)
    trace_logs = [r for r in logs if r.get('method') is None and 'us' in r]
    traces = [{'kid': row['kid'], 'shape': row['shape'], **{k: t[k] for k in ['round', 'label', 'us']}}
              for row in result['rows'] for t in row['timings']]
    assert trace_logs == traces
    epoch = claim_for(name, result)
    argv = epoch['start']['command']['argv']
    plan_path = Path(argv[argv.index('--plan') + 1])
    assert sha(plan_path) == result['plan_sha256'] and read(plan_path) == result['plan']
    assert Path(argv[argv.index('--output') + 1]) == path
    assert argv[argv.index('--repetitions') + 1] == '8'
    assert argv[argv.index('--iters') + 1] == str(expected_iters)
    return result, {'result': str(path.relative_to(ROOT)), 'result_sha256': sha(path),
                    'log_sha256': sha(log_path), 'trace_log_matches': True,
                    'claim_epoch': epoch, 'recorded_plan_content_and_sha_match': True,
                    'plan_sha256': result['plan_sha256'],
                    'runner_sha256': result['runner_sha256'], 'libraries': result['libraries'],
                    'gpu': result['gpu'], 'started': result['started'], 'finished': result['finished']}, logs


def event_audit(name):
    result, record, logs = check_base(name, 51)
    assert result['event_confirmation_requested']
    assert len(result['rows']) == 1
    row = result['rows'][0]
    event = row['event_confirmation']
    assert event['status'] == 'passed' and event['rounds'] == 5
    assert event['iters_per_graph'] == 51 and event['shared_pool']
    assert event['rotation']['count'] == 51 and len(event['pool_pointers']) == 51
    assert all(p['workspace'] == 0 for p in event['pool_pointers'])
    measurements = event['measurements']
    assert len(measurements) == 10
    expected_logs = []
    checks_count = 0
    for t in measurements:
        assert t['iters'] == 51 and t['rotation_count'] == 51
        assert t['us_per_call'] == t['event_total_ms'] * 1000 / 51
        assert [c['pool_index'] for c in t['all_pool_checks']] == list(range(51))
        for c in t['all_pool_checks']:
            assert c['errRatio'] == 0 and c['output_repeatable']
            assert c['output_guards'] and c['workspace_guards']
            # No workspace exists. workspace_repeatable=False is not a failure.
            checks_count += 1
        expected_logs.append({'method': 'graph_event_confirmation', 'kid': row['kid'],
                              'shape': row['shape'], **{k: t[k] for k in
                              ['round', 'label', 'event_total_ms', 'iters', 'us_per_call', 'rotation_count']}})
    assert [r for r in logs if r.get('method') == 'graph_event_confirmation'] == expected_logs
    times = {label: [t['us_per_call'] for t in measurements if t['label'] == label]
             for label in ['baseline', 'candidate']}
    mids = {label: median(vals) for label, vals in times.items()}
    assert mids == event['median_us']
    pairs = []
    for rnd in range(5):
        pair = [t for t in measurements if t['round'] == rnd]
        order = ['baseline', 'candidate'] if rnd % 2 == 0 else ['candidate', 'baseline']
        assert [t['label'] for t in pair] == order
        assert all(t['order'] == order for t in pair)
        values = {t['label']: t['us_per_call'] for t in pair}
        pairs.append({'round': rnd, 'order': order, 'baseline_us': values['baseline'],
                      'candidate_us': values['candidate'],
                      'change_percent': 100 * (values['baseline'] / values['candidate'] - 1)})
    shape = tuple(row['shape'])
    inv = SHAPES[shape]
    assert inv['parent_id'] == row['kid']
    bm = 128 if row['kid'] == 9021 else 160
    wg = ((shape[0] + bm - 1) // bm) * ((shape[1] + 127) // 128)
    assert inv['producer_workgroups'] == wg
    record.update(target_index=row['target_index'], kid=row['kid'], shape=list(shape),
                  current745_same_parent_winner=True, test_grid_workgroups=wg,
                  actual_branch=inv['actual_branch_label'], original_correctness=row['correctness'],
                  event_log_matches10=True, event_pool_checks_count=checks_count,
                  event_pool_checks_passed=True, workspace_present=False,
                  event_median_baseline_us=mids['baseline'], event_median_candidate_us=mids['candidate'],
                  event_ratio_of_medians_change_percent=100 * (mids['baseline'] / mids['candidate'] - 1),
                  paired_rounds=pairs, paired_positive_rounds=sum(p['change_percent'] > 0 for p in pairs),
                  median_paired_change_percent=median(p['change_percent'] for p in pairs),
                  baseline_range_over_median_percent=100 * (max(times['baseline']) - min(times['baseline'])) / mids['baseline'],
                  candidate_range_over_median_percent=100 * (max(times['candidate']) - min(times['candidate'])) / mids['candidate'],
                  trace_change_percent=100 * (row['median_us']['baseline'] / row['median_us']['candidate'] - 1))
    return record


def main():
    mechanism, mech, _ = check_base('compute_prologue_mechanism', 11)
    assert len(mechanism['rows']) == 32 and not mechanism['event_confirmation_requested']
    mech.update(rows=32, baseline_candidate_original_checks=32 * 2 * 8,
                numerical_repetitions_per_side=8, event_measured=False,
                timing_interpretation='one profiler round of 11 iterations; screening only',
                coverage_targets=mechanism['plan']['targets'],
                all_numerical_repeatability_guards_passed=True)
    rows = [event_audit('compute_true_winner_repeat')]
    rows.extend(event_audit(f'compute_confirmation_{i}') for i in range(6))
    final_path = OUT / 'results' / 'compute_confirmation_6.json'
    latest = read(final_path)
    if latest['status'] == 'passed' and not latest.get('contamination'):
        rows.append(event_audit('compute_confirmation_6'))
        pending = []
    else:
        pending = [{'target_index': 6, 'kid': 9022, 'shape': [480, 65536, 1536],
                    'status': latest['status'], 'result': str(final_path.relative_to(ROOT)),
                    'result_sha256': sha(final_path), 'contamination': latest.get('contamination'),
                    'timings_usable_for_final_decision': False}]
    excluded = []
    for path in sorted((OUT / 'results').glob('compute_confirmation_6.attempt_*.json')):
        d = read(path)
        excluded.append({'result': str(path.relative_to(ROOT)), 'sha256': sha(path),
                         'status': d['status'], 'contamination': d.get('contamination'),
                         'timings_usable_for_final_decision': False})
    report = {'status': 'cpu_audit_passed_no_new_gpu_execution', 'new_gpu_execution': False,
              'production_sources_modified': False, 'source_head': INVENTORY['source_head'],
              'mechanism_correctness': mech, 'event_rows': rows,
              'pending_confirmation': pending, 'excluded_contaminated_attempts': excluded,
              'summary': {'clean_event_shapes': len(rows), 'all_same_parent_current745_winners': True,
                          'all_paired_rounds_positive': all(r['paired_positive_rounds'] == 5 for r in rows),
                          'event_output_checks': sum(r['event_pool_checks_count'] for r in rows),
                          'global_adoption': False},
              'limits': ['Mechanism cases do not cover the complete 735/691 supported candidate domains.',
                         'Winner representatives do not cover all 42/159 current winners.',
                         'Event claims concern complete private launches; no counter/ATT proof of prologue root cause.',
                         'No physical clock, power or thermal lock; clean process monitoring does not establish fixed clocks.',
                         'Profiler single-round screen and shared-pool Event use distinct timing protocols.']}
    target = OUT / 'results' / 'compute_prologue_confirmation_analysis.json'
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'output': str(target), 'mechanism': 32, 'clean_events': len(rows),
                      'pending': len(pending), 'event_gains_percent': [r['event_ratio_of_medians_change_percent'] for r in rows]}))


if __name__ == '__main__':
    main()
