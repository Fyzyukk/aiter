#!/usr/bin/env python3
"""CPU-only strict audit of a scale_issue_publish screen/confirmation result."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent.parent / 'opus_bound_analysis_20261007'
LABELS = ['baseline', 'candidate']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def log_rows(path):
    rows = []
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def require(condition, message):
    if not condition:
        raise ValueError(message)


def arg_value(argv, option):
    require(argv.count(option) == 1, f'Expected one {option} argument')
    return argv[argv.index(option) + 1]


def find_epoch(claims, result, input_path):
    epochs = []
    claimed = None
    active = None
    for row in claims:
        event = row.get('event')
        if event == 'claimed':
            claimed = row
        elif event == 'start':
            command = row['command']
            if '--output' in command['argv'] and Path(arg_value(command['argv'], '--output')).resolve() == input_path:
                require(active is None, 'Overlapping target command starts')
                active = {'claimed': claimed, 'start': row, 'owner_identity': [], 'monitor': []}
        elif active and event == 'owner_identity' and row['name'] == active['start']['command']['name']:
            active['owner_identity'].append(row)
        elif active and event == 'monitor':
            active['monitor'].append(row)
        elif active and event == 'end' and row['name'] == active['start']['command']['name']:
            active['end'] = row
            epochs.append(active)
            active = None
    selected = [e for e in epochs if e['start']['time'] <= result['started'] <
                result['finished'] <= e['end']['time']]
    require(len(selected) == 1, 'Result must match exactly one completed command epoch')
    epoch = selected[0]
    require(epoch['end']['returncode'] == 0 and epoch['end']['contamination'] is False, 'Unclean command epoch')
    require(epoch['claimed'] is not None, 'Missing physical GPU claim')
    gpu = result['gpu']
    require(epoch['claimed']['gpu']['bdf'] == gpu['pci_bdf'] == gpu['expected_pci_bdf'], 'Physical GPU mismatch')
    require(str(epoch['claimed']['gpu']['hip_index']) == gpu['HIP_VISIBLE_DEVICES'], 'Visible HIP index mismatch')
    require(len(epoch['owner_identity']) == 1, 'Missing/ambiguous owner handshake')
    owner = epoch['owner_identity'][0]
    require(owner['new_host_pids'] == [owner['host_pid']], 'Owner host PID is not uniquely resolved')
    require(epoch['start']['time'] <= owner['time'] <= epoch['end']['time'], 'Owner handshake outside epoch')
    require(sha(OLD / 'owned_python_launch.py') == owner['launcher_sha256'], 'Owner launcher SHA mismatch')
    require(all(m['child_pid'] == owner['inner_pid'] and
                all(p['pid'] == owner['host_pid'] for p in m['processes'])
                for m in epoch['monitor']), 'Contaminated or mismatched monitor')
    argv = epoch['start']['command']['argv']
    require('--event-confirm' in argv, 'Not an Event confirmation command')
    require(arg_value(argv, '--rounds') == '5' and arg_value(argv, '--iters') == '51' and
            arg_value(argv, '--repetitions') == '8', 'Command protocol mismatch')
    return epoch


def identities(result):
    source = read(ROOT / 'source_manifest.json')
    build = read(ROOT / 'build_manifest.json')
    device = read(ROOT / 'device_audit.json')
    order = read(ROOT / 'isa_order_audit.json')
    require(build['status'] == device['status'] == 'passed', 'Build/device audit failed')
    require(order['device_audit_sha256'] == sha(ROOT / 'device_audit.json'), 'ISA order audit does not match device audit')
    built = {r['side']: r for r in build['builds']}
    audited = {r['label']: r for r in device['libraries']}
    require(set(result['libraries']) == set(LABELS), 'Unexpected result libraries')
    for label in LABELS:
        path = (ROOT / label / 'experiments.so').resolve()
        recorded = result['libraries'][label]
        require(Path(recorded['path']).resolve() == path, 'Unexpected tested library path')
        require(sha(path) == recorded['sha256'] == built[label]['library_sha256'] ==
                audited[label]['input_sha256'], 'Test/build/audit library SHA mismatch')
        require(built[label]['returncode'] == 0, 'Library build failed')
        for item in source['source_files']:
            require(sha(ROOT / label / item['path']) == item[label + '_sha256'], 'Source manifest mismatch')
        for kernel in audited[label]['kernels']:
            pc_path = ROOT / 'device_audit_artifacts' / label / f"kid{kernel['kid']}.s"
            if kernel['kid'] == 9021:
                require(sha(pc_path) == order['evidence'][label]['source_sha256'], 'ISA artifact SHA mismatch')
    require(result['runner_sha256'] == sha(OLD / 'experiment_runner.py'), 'Runner SHA mismatch')
    require(result['source_head'] == read(OLD / 'shape_inventory.json')['source_head'], 'Source HEAD mismatch')
    return {name: {'path': str((ROOT / name).resolve()), 'sha256': sha(ROOT / name)} for name in
            ('source_manifest.json', 'build_manifest.json', 'device_audit.json', 'isa_order_audit.json')}


def audit_row(row, target, result):
    require(row['kid'] == target['kid'] == 9021 and row['shape'] == target['shape'], 'Target row mismatch')
    require(target['signed'] is True and target['seed'] == 17 and row['split'] == 1, 'Numerical target contract mismatch')
    for label in LABELS:
        check = row['correctness'][label]
        require(check['repetitions'] == 8 and check['errRatio'] == 0 and check['repeatable'] and
                check['output_guards'] and check['workspace_guards'], 'Original numerical/guard/repeatability failure')
    require(len(row['timings']) == 10, 'Expected five paired trace rounds')
    for round_index in range(5):
        pair = row['timings'][2 * round_index:2 * round_index + 2]
        order = LABELS if round_index % 2 == 0 else LABELS[::-1]
        require([t['label'] for t in pair] == order and all(t['round'] == round_index for t in pair), 'Trace AB/BA ordering mismatch')
        require(all(t['errRatio'] == 0 and t['output_guards'] and t['workspace_guards'] and
                    math.isfinite(t['us']) and t['us'] > 0 for t in pair), 'Invalid trace result')
    event = row['event_confirmation']
    require(event['status'] == 'passed' and event['rounds'] == 5 and event['iters_per_graph'] == 51 and
            event['shared_pool'] is True and event['rotation']['count'] == 51, 'Event protocol mismatch')
    pointers = event['pool_pointers']
    require(len(pointers) == 51 and all(p['workspace'] == 0 for p in pointers), 'Invalid Event pool/workspace')
    require(all(len({p[k] for p in pointers}) == 51 and all(p[k] > 0 for p in pointers)
                for k in ('a', 'b', 'c', 'sfa', 'sfb')), 'Event input/output pools do not have 51 unique valid pointers')
    measurements = event['measurements']
    require(len(measurements) == 10, 'Incomplete Event rounds')
    pairs = []
    for round_index in range(5):
        pair = measurements[2 * round_index:2 * round_index + 2]
        order = LABELS if round_index % 2 == 0 else LABELS[::-1]
        require([t['label'] for t in pair] == order and all(t['round'] == round_index and
                t['order'] == order for t in pair), 'Event AB/BA ordering mismatch')
        for t in pair:
            require(t['iters'] == t['rotation_count'] == 51 and math.isfinite(t['event_total_ms']) and
                    t['event_total_ms'] > 0 and t['us_per_call'] == t['event_total_ms'] * 1000 / 51,
                    'Event units or pool size mismatch')
            checks = t['all_pool_checks']
            require([c['pool_index'] for c in checks] == list(range(51)), 'Incomplete Event pool checks')
            require(all(c['errRatio'] == 0 and c['output_repeatable'] and c['output_guards'] and
                        c['workspace_guards'] for c in checks), 'Event numerical/guard/repeatability failure')
        times = {t['label']: t['us_per_call'] for t in pair}
        pairs.append({'round': round_index, 'order': order, **{label + '_us': times[label] for label in LABELS},
                      'baseline_over_candidate_speedup': times['baseline'] / times['candidate'],
                      'speedup_percent': 100 * (times['baseline'] / times['candidate'] - 1),
                      'candidate_time_change_percent': 100 * (times['candidate'] / times['baseline'] - 1)})
    times = {label: [t['us_per_call'] for t in measurements if t['label'] == label] for label in LABELS}
    medians = {label: statistics.median(values) for label, values in times.items()}
    require(medians == event['median_us'], 'Stored Event medians mismatch')
    ratio = medians['baseline'] / medians['candidate']
    require(event['median_speedup']['candidate'] == ratio, 'Stored Event speedup mismatch')
    return {'target_index': row['target_index'], 'kid': 9021, 'shape': row['shape'],
            'event_median_us': medians, 'baseline_over_candidate_speedup': ratio,
            'speedup_percent': 100 * (ratio - 1),
            'candidate_time_change_percent': 100 * (1 / ratio - 1), 'paired_rounds': pairs,
            'candidate_faster_rounds': sum(p['speedup_percent'] > 0 for p in pairs),
            'candidate_slower_rounds': sum(p['speedup_percent'] < 0 for p in pairs),
            'median_paired_speedup_percent': statistics.median(p['speedup_percent'] for p in pairs),
            'range_over_median_percent': {label: 100 * (max(values) - min(values)) / medians[label]
                                          for label, values in times.items()},
            'numerical_calls_original': 16, 'event_pool_output_checks': 510,
            'workspace_present': False, 'all_checks_passed': True}


def aggregate(rows):
    if not rows:
        return None
    geo = math.exp(statistics.mean(math.log(r['baseline_over_candidate_speedup']) for r in rows))
    sums = {label: sum(r['event_median_us'][label] for r in rows) for label in LABELS}
    return {'shape_count': len(rows), 'geomean_speedup': geo, 'geomean_speedup_percent': 100 * (geo - 1),
            'sum_of_shape_medians_us': sums, 'sum_time_speedup_percent': 100 * (sums['baseline'] / sums['candidate'] - 1),
            'median_faster_shapes': sum(r['speedup_percent'] > 0 for r in rows),
            'median_slower_shapes': sum(r['speedup_percent'] < 0 for r in rows),
            'five_of_five_faster_shapes': sum(r['candidate_faster_rounds'] == 5 for r in rows),
            'five_of_five_slower_shapes': sum(r['candidate_slower_rounds'] == 5 for r in rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--claim', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    input_path = args.input.resolve()
    output = args.output.resolve()
    require(output != input_path and not output.exists(), 'Output must be a new analysis file')
    result = read(input_path)
    require(result['status'] == 'passed' and result['profiling_only'] is False and
            result['event_confirmation_requested'] and result['rounds'] == 5 and result['iters'] == 51,
            'Input is not a complete five-round Event result')
    audits = identities(result)
    epoch = find_epoch(log_rows(args.claim), result, input_path)
    plan_path = Path(arg_value(epoch['start']['command']['argv'], '--plan'))
    require(sha(plan_path) == result['plan_sha256'] and read(plan_path) == result['plan'], 'Plan content/SHA mismatch')
    require(result['plan']['workspace'] is False, 'Unexpected workspace')
    require(result['plan']['libraries'] == {label: result['libraries'][label]['path'] for label in LABELS}, 'Plan library paths mismatch')
    expected_indices = list(range(len(result['plan']['targets']))) if result['target_index'] is None else [result['target_index']]
    require([r['target_index'] for r in result['rows']] == expected_indices, 'Incomplete or unordered result target coverage')
    rows = [audit_row(row, result['plan']['targets'][row['target_index']], result) for row in result['rows']]
    require(len({tuple(r['shape']) for r in rows}) == len(rows), 'Duplicate shape in result')
    logs = log_rows(input_path.with_suffix('.log'))
    trace_logs = [r for r in logs if r.get('method') is None and 'us' in r]
    expected_trace = [{'kid': r['kid'], 'shape': r['shape'], **{key: t[key] for key in ('round', 'label', 'us')}}
                      for r in result['rows'] for t in r['timings']]
    event_logs = [r for r in logs if r.get('method') == 'graph_event_confirmation']
    expected_event = [{'method': 'graph_event_confirmation', 'kid': r['kid'], 'shape': r['shape'],
                       **{key: t[key] for key in ('round', 'label', 'event_total_ms', 'iters', 'us_per_call', 'rotation_count')}}
                      for r in result['rows'] for t in r['event_confirmation']['measurements']]
    require(trace_logs == expected_trace and event_logs == expected_event, 'Result/Event log mismatch')
    parts = sorted((OLD / 'plans/compute_prologue_full_scope').glob('compute_prologue_winners_9021_part*.json'))
    expected_winners = {tuple(t['shape']) for p in parts for t in read(p)['targets']}
    require(len(expected_winners) == 42, 'Frozen winner set mismatch')
    actual_shapes = {tuple(r['shape']) for r in rows}
    winner_rows = [r for r in rows if tuple(r['shape']) in expected_winners]
    for r in rows:
        r['current745_same_parent_winner'] = tuple(r['shape']) in expected_winners
    report = {'status': 'strict_cpu_Event_audit_passed', 'generated_utc': datetime.now(timezone.utc).isoformat(),
              'new_gpu_execution': False, 'input': {'path': str(input_path), 'sha256': sha(input_path)},
              'log': {'path': str(input_path.with_suffix('.log')), 'sha256': sha(input_path.with_suffix('.log'))},
              'claim_log': {'path': str(args.claim.resolve()), 'sha256': sha(args.claim)},
              'plan': {'path': str(plan_path), 'sha256': sha(plan_path)}, 'audit_identity': audits,
              'runner_sha256': result['runner_sha256'], 'libraries': result['libraries'],
              'gpu': result['gpu'], 'claim_epoch': epoch, 'started': result['started'], 'finished': result['finished'],
              'rows': rows, 'all_targets_summary': aggregate(rows), 'actual_winner_summary': aggregate(winner_rows),
              'winner_coverage': {'expected': 42, 'covered': len(winner_rows),
                                  'complete_exact42': len(winner_rows) == 42,
                                  'remaining': [list(s) for s in sorted(expected_winners - actual_shapes)],
                                  'nonwinner_targets': [list(s) for s in sorted(actual_shapes - expected_winners)],
                                  'frozen_inputs': {str(p): sha(p) for p in parts}},
              'numerical_calls_original': len(rows) * 16, 'event_pool_output_checks': len(rows) * 510,
              'trace_and_Event_logs_exactly_match': True, 'adopted': False,
              'limits': ['Five paired rounds are from one clean process window; no fixed-clock or independent-window significance claim.',
                         'Event medians/paired ratios are complete private calls. Profiler screening values do not decide retention.',
                         'Geomean gives equal weight to measured shapes; sumtime is a synthetic sum of medians, not a measured batch.',
                         'No evidence here determines a unique hardware bottleneck or extends performance to unmeasured supports.']}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': report['status'], 'targets': len(rows), 'winner_covered': len(winner_rows),
                      'winner_complete': report['winner_coverage']['complete_exact42'],
                      'speedup_percent': [r['speedup_percent'] for r in rows], 'output': str(output)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
