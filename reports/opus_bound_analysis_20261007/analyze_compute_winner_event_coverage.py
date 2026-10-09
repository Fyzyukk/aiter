#!/usr/bin/env python3
"""CPU audit and exact deduplication of complete current-winner prologue Events.

Reads all original and new result files, validates their own claim epochs,
plans, binaries, signed8 checks and 51-call/5-round shared-pool Event protocol.
Creates a new remaining queue without modifying any original plan or queue.
"""
import argparse
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


def logs(path):
    result = []
    for line in path.read_text().splitlines():
        try:
            v = json.loads(line)
        except ValueError:
            continue
        if isinstance(v, dict):
            result.append(v)
    return result


def require(value, message):
    if not value:
        raise ValueError(message)


def key(kid, shape):
    return (kid, *shape)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary-only', action='store_true',
                        help='write coverage report only; never rewrite an active remaining queue')
    args = parser.parse_args()
    source_queue_path = OUT / 'compute_prologue_all_winners_event_short_k_queue.json'
    source_queue = read(source_queue_path)
    inventory = read(OUT / 'shape_inventory.json')
    inv = {key(r['parent_id'], [r['M'], r['N'], r['K']]): r
           for r in inventory['rows'] if r['parent_id'] in [9021, 9022]}
    commands = {}
    for command in source_queue['commands']:
        argv = command['argv']
        plan = read(Path(argv[argv.index('--plan') + 1]))
        target = plan['targets'][int(argv[argv.index('--target-index') + 1])]
        shape_key = key(target['kid'], target['shape'])
        require(shape_key in inv and shape_key not in commands, 'winner source queue shape mismatch')
        commands[shape_key] = command
    require(len(commands) == len(inv) == 201, 'expected complete 201 current winner domain')
    libraries = {label: {'path': str(OUT / 'compute_prologue' / label / 'experiments.so'),
                         'sha256': sha(OUT / 'compute_prologue' / label / 'experiments.so')}
                 for label in ['baseline', 'candidate']}
    runner_sha = sha(OUT / 'experiment_runner.py')
    claims = logs(OUT / 'gpu_claim_log.jsonl')
    clean = {}
    excluded = []
    for path in sorted((OUT / 'results').glob('*.json')):
        try:
            result = read(path)
        except (OSError, ValueError):
            continue
        if not isinstance(result, dict) or not isinstance(result.get('rows'), list):
            continue
        relevant = [r for r in result['rows'] if isinstance(r, dict) and r.get('kid') in [9021, 9022]
                    and isinstance(r.get('shape'), list) and key(r['kid'], r['shape']) in commands
                    and 'event_confirmation' in r]
        if not relevant:
            continue
        record = {'result': str(path.relative_to(ROOT)), 'sha256': sha(path),
                  'application_status': result.get('status'), 'included_in_coverage': False}
        if '.attempt_' in path.name or result.get('status') != 'passed' or result.get('contamination'):
            record.update(reason='archived, incomplete or contaminated result', contamination=result.get('contamination'))
            excluded.append(record)
            continue
        try:
            require(result['libraries'] == libraries, 'result library paths/SHA differ from current prologue pair')
            require(result['source_head'] == inventory['source_head'], 'source HEAD mismatch')
            require(result['runner_sha256'] == runner_sha, 'result runner differs from current tested runner')
            require(result['event_confirmation_requested'] and not result['profiling_only'] and result['iters'] == 51,
                    'Event confirmation must use 51 calls')
            require(result.get('finished') and result['finished'] > result['started'], 'missing finished epoch')
            starts = [c for c in claims if c.get('event') == 'start'
                      and c.get('command', {}).get('name') == path.stem and c['time'] <= result['started']]
            require(starts, 'root start missing')
            start = max(starts, key=lambda c: c['time'])
            ends = [c for c in claims if c.get('event') == 'end' and c.get('name') == path.stem
                    and c['time'] >= result['finished']]
            require(ends, 'root end missing')
            end = min(ends, key=lambda c: c['time'])
            require(end['returncode'] == 0 and end['contamination'] is False, 'root end not clean')
            claim = max((c for c in claims if c.get('event') == 'claimed' and c['time'] <= start['time']),
                        key=lambda c: c['time'])
            require(not any(c.get('event') == 'claimed' and start['time'] < c['time'] < end['time'] for c in claims),
                    'physical claim changed during command')
            owners = [c for c in claims if c.get('event') == 'owner_identity'
                      and c.get('name') == path.stem and start['time'] <= c['time'] <= end['time']]
            owner = None
            if owners:
                require(len(owners) == 1, 'multiple owner identity markers in own command epoch')
                owner = owners[0]
                require(isinstance(owner['host_pid'], int) and isinstance(owner['inner_pid'], int)
                        and owner['host_pid'] > 0 and owner['inner_pid'] > 0
                        and owner['new_host_pids'] == [owner['host_pid']]
                        and owner['host_pid'] not in owner['baseline_host_pids'],
                        'owner marker does not identify one newly registered host PID')
                require(owner['method'] == 'KFD open + unique sysfs registration + same-mm runpy handshake',
                        'owner identity method differs')
                require(owner['launcher_sha256'] == sha(OUT / 'owned_python_launch.py'),
                        'owner launcher changed since marker')
                monitors = [c for c in claims if c.get('event') == 'monitor'
                            and owner['time'] <= c['time'] <= end['time']]
                require(monitors and all(c['child_pid'] == owner['inner_pid'] for c in monitors),
                        'owner inner PID differs from own monitor epoch')
            gpu = result['gpu']
            require(gpu['pci_bdf'] == gpu['expected_pci_bdf'] == claim['gpu']['bdf'], 'physical GPU/claim mismatch')
            require(gpu['HIP_VISIBLE_DEVICES'] == str(claim['gpu']['hip_index']), 'HIP ordinal/claim mismatch')
            argv = start['command']['argv']
            require(Path(argv[argv.index('--output') + 1]) == path, 'claim output mismatch')
            require('--event-confirm' in argv and argv[argv.index('--iters') + 1] == '51'
                    and argv[argv.index('--repetitions') + 1] == '8', 'claimed protocol differs')
            plan_path = Path(argv[argv.index('--plan') + 1])
            require(sha(plan_path) == result['plan_sha256'] and read(plan_path) == result['plan'], 'recorded plan changed')
            require(result['plan']['workspace'] is False, 'prologue event must not have workspace')
            selected_index = int(argv[argv.index('--target-index') + 1]) if '--target-index' in argv else None
            expected_indices = [selected_index] if selected_index is not None else list(range(len(result['plan']['targets'])))
            require(result['target_index'] == selected_index
                    and [r['target_index'] for r in result['rows']] == expected_indices,
                    'single-target or batch executed indices differ from claimed command')
            require(len(relevant) == len(result['rows']), 'result contains rows outside current-winner prologue domain')
            log_path = Path(start['command']['log'])
            event_logs = [r for r in logs(log_path) if r.get('method') == 'graph_event_confirmation']
            for row in relevant:
                target = result['plan']['targets'][row['target_index']]
                require(target['kid'] == row['kid'] and target['shape'] == row['shape'] and target['signed'] is True,
                        'row/target identity or signed input mismatch')
                require(row['split'] == 1, 'unexpected split')
                for label in ['baseline', 'candidate']:
                    check = row['correctness'][label]
                    require(check['repetitions'] == 8 and check['errRatio'] == 0 and check['repeatable']
                            and check['output_guards'] and check['workspace_guards'], 'signed8 original checks incomplete')
                e = row['event_confirmation']
                require(e['status'] == 'passed' and e['rounds'] == 5 and e['iters_per_graph'] == 51 and e['shared_pool'],
                        'shared-pool/51-call/5-round Event incomplete')
                measurements = e['measurements']
                require(len(measurements) == 10, 'ten Event measurements required')
                pool_count = e['rotation']['count']
                require(pool_count >= 1 and len(e['pool_pointers']) == pool_count,
                        'shared pool identity count differs')
                require(all(p['workspace'] == 0 for p in e['pool_pointers']), 'prologue shared pool has workspace')
                expected_log = []
                check_count = 0
                pairs = []
                times = {'baseline': [], 'candidate': []}
                for rnd in range(5):
                    pair = [m for m in measurements if m['round'] == rnd]
                    order = ['baseline', 'candidate'] if rnd % 2 == 0 else ['candidate', 'baseline']
                    require([m['label'] for m in pair] == order, 'AB/BA Event order mismatch')
                    values = {}
                    for m in pair:
                        require(m['order'] == order and m['iters'] == 51 and m['rotation_count'] == pool_count,
                                'Event order/iterations/pool differs')
                        require(math.isfinite(m['us_per_call']) and m['us_per_call'] > 0
                                and m['us_per_call'] == m['event_total_ms'] * 1000 / 51, 'Event unit mismatch')
                        require([c['pool_index'] for c in m['all_pool_checks']] == list(range(pool_count)), 'all pool checks missing')
                        for c in m['all_pool_checks']:
                            require(c['errRatio'] == 0 and c['output_repeatable'] and c['output_guards']
                                    and c['workspace_guards'], 'Event pool numerical/repeatability/guard failure')
                            check_count += 1
                        expected_log.append({'method': 'graph_event_confirmation', 'kid': row['kid'], 'shape': row['shape'],
                            **{k: m[k] for k in ['round', 'label', 'event_total_ms', 'iters', 'us_per_call', 'rotation_count']}})
                        times[m['label']].append(m['us_per_call'])
                        values[m['label']] = m['us_per_call']
                    pairs.append({'round': rnd, 'order': order, 'baseline_us': values['baseline'],
                                  'candidate_us': values['candidate'],
                                  'change_percent': 100 * (values['baseline'] / values['candidate'] - 1)})
                actual_logs = [v for v in event_logs if v['kid'] == row['kid'] and v['shape'] == row['shape']]
                require(actual_logs == expected_log, 'ten Event logs do not match JSON')
                mids = {label: median(v) for label, v in times.items()}
                require(mids == e['median_us'], 'Event median differs')
                require(e['median_speedup']['candidate'] == mids['baseline'] / mids['candidate'], 'Event ratio differs')
                shape_key = key(row['kid'], row['shape'])
                clean.setdefault(shape_key, []).append({'result': str(path.relative_to(ROOT)), 'result_sha256': sha(path),
                    'log_sha256': sha(log_path), 'plan_sha256': result['plan_sha256'], 'runner_sha256': result['runner_sha256'],
                    'parent_id': row['kid'], 'shape': row['shape'], 'seed': target.get('seed', 17), 'signed': True,
                    'claim_epoch': {'claim': claim, 'start': start, 'end': end}, 'gpu': gpu,
                    'owner_identity': owner,
                    'owner_identity_audit': ('unique_new_host_PID_marker_validated' if owner else 'legacy_clean_claim_no_owner_marker'),
                    'original_repetitions_per_side': 8, 'shared_pool': True, 'pool_count': pool_count,
                    'event_pool_checks_passed': True, 'event_pool_checks_count': check_count,
                    'event_rounds': 5, 'calls_per_graph': 51, 'event_median_us': mids,
                    'event_ratio_of_medians_change_percent': 100 * (mids['baseline'] / mids['candidate'] - 1),
                    'paired_rounds': pairs, 'paired_positive_rounds': sum(p['change_percent'] > 0 for p in pairs),
                    'baseline_range_over_median_percent': 100 * (max(times['baseline']) - min(times['baseline'])) / mids['baseline'],
                    'candidate_range_over_median_percent': 100 * (max(times['candidate']) - min(times['candidate'])) / mids['candidate']})
        except (KeyError, ValueError, OSError, TypeError) as exc:
            record['reason'] = str(exc)
            excluded.append(record)
    remaining = {'env': source_queue['env'], 'commands': [c for shape_key, c in commands.items() if shape_key not in clean]}
    remaining_path = OUT / 'compute_prologue_winners_remaining_event_queue.json'
    if not args.summary_only:
        remaining_path.write_text(json.dumps(remaining, indent=2) + '\n')
    covered_rows = []
    for shape_key, evidence in clean.items():
        evidence.sort(key=lambda r: r['claim_epoch']['end']['time'])
        # All accepted repeats are preserved; latest clean round is the summary.
        covered_rows.append({'parent_id': shape_key[0], 'shape': list(shape_key[1:]),
                             'clean_records': evidence, 'summary_record': evidence[-1]})
    covered_rows.sort(key=lambda r: (r['parent_id'], r['shape']))
    summary = {'expected_winner_shapes': 201, 'clean_complete_unique_winner_shapes': len(clean),
               'clean_Event_records_including_repeats': sum(len(v) for v in clean.values()),
               'remaining_unique_winner_shapes': len(remaining['commands']),
               'parent_covered_counts': dict(Counter(k[0] for k in clean)),
               'parent_remaining_counts': dict(Counter(k[0] for k in commands if k not in clean)),
               'full_201_Event_coverage_complete': len(clean) == 201, 'adopted': False}
    report = {'status': 'complete_201_Event_coverage' if len(clean) == 201 else 'partial_201_Event_coverage',
              'new_gpu_execution': False, 'production_sources_modified': False, 'source_head': inventory['source_head'],
              'libraries': libraries, 'summary': summary, 'covered_winners': covered_rows,
              'remaining_winners': [{'parent_id': k[0], 'shape': list(k[1:]), 'command': c['name']}
                                    for k, c in commands.items() if k not in clean],
              'excluded_records': excluded, 'source_queue_sha256': sha(source_queue_path),
              'remaining_queue': str(remaining_path.relative_to(ROOT)),
              'remaining_queue_sha256': sha(remaining_path) if remaining_path.exists() else None,
              'remaining_queue_written': not args.summary_only,
              'remaining_queue_may_include_newly_completed_commands': args.summary_only,
              'limits': ['Each unique current parent+shape is counted once; accepted independent repeats are retained, not counted as new shape coverage.',
                         'Only matching current private binaries, signed8 checks, own clean claim epoch and full 5-round shared-pool 51-call Event count.',
                         'Counters/profiler trace activity do not replace Event coverage; this report does not itself adopt an optimization.',
                         'No requirement to finish all 1426 supported fallback screens; incomplete support remains separately partial.',
                         'Rerun CPU summary after a completed batch; do not rewrite the remaining queue during active execution.',
                         'New same-mm KFD owner markers require one newly registered host PID and matching command/monitor epoch. Nonce equality was checked by the parent launcher, not exposed in logs; CPU summary checks the recorded marker.',
                         'Earlier clean epochs without owner markers retain their recorded claim evidence; interrupted or contaminated attempts remain excluded.']}
    report_path = OUT / 'results/compute_prologue_winner_event_coverage.json'
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    require('torch' not in sys.modules, 'CPU audit must not import torch')
    print(json.dumps({'output': str(report_path), 'remaining_queue': str(remaining_path), 'summary': summary,
                      'excluded_records': len(excluded)}))


if __name__ == '__main__':
    main()
